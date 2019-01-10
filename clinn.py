import itertools
from functools import reduce
import operator
import tensorflow as tf
import time
import random
import layers
import math
import argparse
import parseutils as pu
import utils
import pandas as pd
import numpy as np
from layers import maxclip, fc
from utils import msgtime, str_memusage, print_prog_bar, fcn_stats
import csv

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

# Loading the dataset
dataset = pd.read_csv('Iris_Dataset.csv')

dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding

values = list(dataset.columns.values)

y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')

# Shuffle Data
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

# Creating a Train and a Test Dataset
test_size = 50
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

# Suppress Unecessary Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Function to Build the Parser for CLI
def build_parser():
    parser = argparse.ArgumentParser(description='CLI Utility for NNPSO')

    # Dataset Generation Parameters
    parser.add_argument('--bs', type=pu.intg0, default=32,
                        help='batchsize', metavar='N_BATCHSIZE')
    parser.add_argument('--xorn', type=pu.intg0, default=4,
                        help='Number of XOR Inputs', metavar='N_IN')

    # PSO Parameters
    parser.add_argument('--pno', type=pu.intg0, default=32,
                        help='number of particles', metavar='N_PARTICLES')
    parser.add_argument('--gbest', type=pu.floatnorm, default=0.8,
                        help='global best for PSO', metavar='G_BEST_FACTOR')
    parser.add_argument('--lbest', type=pu.floatnorm, default=0.7,
                        help='local best for PSO', metavar='L_BEST_FACTOR')
    parser.add_argument('--pbest', type=pu.floatnorm, default=0.6,
                        help='local best for PSO', metavar='P_BEST_FACTOR')
    parser.add_argument('--veldec', type=pu.floatnorm, default=1,
                        help='Decay in velocity after each position update',
                        metavar='VELOCITY_DECAY')
    parser.add_argument('--vr', action='store_true',
                        help='Restrict the Particle Velocity')
    parser.add_argument('--mv', type=pu.pfloat, default=0.005,
                        help='Maximum velocity for a particle if restricted',
                        metavar='MAX_VEL')
    parser.add_argument('--mvdec', type=pu.floatnorm, default=1,
                        help='Multiplier for Max Velocity with each update',
                        metavar='MAX_VEL_DECAY')
    # Hyrid Parmeters
    parser.add_argument('--hybrid', action='store_true',
                        help='Use Adam along with PSO')
    parser.add_argument('--lr', type=pu.pfloat, default=0.1,
                        help='Learning Rate if Hybrid Approach',
                        metavar='LEARNING_RATE')
    parser.add_argument('--lbpso', action='store_true',
                        help='Using Local Best Variant of PSO')

    # Other Parameters
    parser.add_argument('--iter', type=pu.intg0, default=int(1e6),
                        help='number of iterations', metavar='N_INTERATIONS')
    parser.add_argument('--hl', nargs='+', type=int,
                        help='hiddenlayers for the network', default=[3, 2])
    parser.add_argument('--pi', type=pu.intg0, default=100,
                        help='Nos iteration for result printing',
                        metavar='N_BATCHSIZE')

    return parser


msgtime('Script Launched\t\t:')
msgtime('Building Parser\t\t:')
parser = build_parser()
msgtime('Parser Built\t\t:')
msgtime('Parsing Arguments\t:')
args = parser.parse_args()
msgtime('Arguments Parsed\t:')
print('Arguments Obtained\t:', vars(args))

# XOR Dataset Params
N_IN = args.xorn
N_BATCHSIZE = args.bs

# PSO params
N_PARTICLES = args.pno
P_BEST_FACTOR = args.pbest
G_BEST_FACTOR = args.gbest
L_BEST_FACTOR = args.lbest
c1=tf.constant(2.05,dtype=None,shape=None,name='c1')
c2=tf.constant(2.05,dtype=None,shape=None,name='c2')
psi=4.1
chi = abs(2.0 / (2.0 - psi - math.sqrt(psi * psi - 4.0 * psi)))
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Velocity Decay specifies the multiplier for the velocity update
VELOCITY_DECAY = args.veldec

# Velocity Restrict is computationally slightly more expensive
VELOCITY_RESTRICT = args.vr
MAX_VEL = args.mv

# Allows to decay the maximum velocity with each update
# Useful if the network needs very fine tuning towards the end
MAX_VEL_DECAY = args.mvdec

# Hybrid Parameters
HYBRID = args.hybrid
LEARNING_RATE = args.lr
LBPSO = args.lbpso

# Other Params
N_ITERATIONS = args.iter
HIDDEN_LAYERS = args.hl
print(HIDDEN_LAYERS)
PRINT_ITER = args.pi

# Basic Neural Network Definition
# Simple feedforward Network
LAYERS = [N_IN] + HIDDEN_LAYERS + [3]
print('Network Structure\t:', LAYERS)


t_VELOCITY_DECAY = tf.constant(value=VELOCITY_DECAY,
                               dtype=tf.float32,
                               name='vel_decay')
t_MVEL = tf.Variable(MAX_VEL,
                     dtype=tf.float32,
                     name='vel_restrict')

net_in = tf.placeholder(dtype=tf.float32,
                        shape=[None, 4],
                        name='net_in')

label = tf.placeholder(dtype=tf.float32,
                       shape=[None, 3],
                       name='net_label')



print('Mem Usage\t\t:', str_memusage(datatype='M'))
msgtime('Building Network\t:')

# MULTI-PARTICLE NEURAL NETS

losses = []
nets = []
pweights = []
pbiases = []
vweights = []
vbiases = []
pfitlist = []
pweightslist=[]
vweightslist=[]
pbiaseslist=[]
vbiaseslist=[]
weightslist=[]
biaseslist=[]
netlist=[]
len_weights=tf.placeholder(dtype=tf.int32,shape=[])
random_values = []
accuracy_updates=[]

# Positional Updates
bias_updates = []
weight_updates = []

# Velocity Updates
vweight_updates = []
vbias_updates = []

# Fitness Updates
fit_updates = []

# Control Updates - Controling PSO inside tf.Graph
control_updates = []

# Hybrid Updates - Using of PSO + Traditional Approaches
hybrid_updates = []

# Global Best
gweights = []
gbiases = []
gfit = tf.Variable(math.inf, name='gbestfit')

# Local Best
lweights= []
lbiases =[]
lfitlist=[]
lweightslist=[]
lbiaseslist=[]
lbestindex = tf.Variable(math.inf, name='lbestindex')

for pno in range(N_PARTICLES):
    weights = []
    biases = []
    pweights = []
    pbiases = []
    pbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=P_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'pbestrand')
    if not LBPSO:
        gbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=G_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'gbestrand')
 
    # Append the random values so that the initializer can be called again
    random_values.append(pbestrand)
    if not LBPSO:
        random_values.append(gbestrand)
    pfit = tf.Variable(math.inf, name='pno' + str(pno + 1) + 'fit')
    net = net_in
    # Define the parameters

    for idx, num_neuron in enumerate(LAYERS[1:]):
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        net, pso_tupple = fc(input_tensor=net,
                             n_output_units=num_neuron,
                             activation_fn='sigmoid',
                             scope=layer_scope,
                             uniform=True)
        w, b, pw, pb, vw, vb = pso_tupple
        vweights.append(vw)
        vbiases.append(vb)
        vweightslist.append(pw)
        vbiaseslist.append(pb)
        weights.append(w)
        weightslist.append(w)
        biaseslist.append(b)
        biases.append(b)
        pweights.append(pw)
        pbiases.append(pb)
        pweightslist.append(pw)
        pbiaseslist.append(pb)
        netlist.append(net)
        lw = tf.Variable(tf.random_uniform(
                shape=[LAYERS[idx],LAYERS[idx+1]],
                dtype=tf.float32),
                name='lw')
        lb = tf.Variable(tf.random_uniform(
                shape=[LAYERS[idx+1]],
                dtype=tf.float32),
                name='lb')
        lweightslist.append(lw)
        lbiaseslist.append(lb)
        if not LBPSO:
            nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
            nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

            # Differences between Particle Best & Current
            pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
            pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)
            # Define & Reuse the GBest
            gw = None
            gb = None
            with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
                gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                 shape=[LAYERS[idx], LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

                gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                 shape=[LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

            # If first Particle add to Global Else it is already present

            if pno == 0:
                gweights.append(gw)
                gbiases.append(gb)

            # Differences between Global Best & Current
            gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
            gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)
            vweight_update = None
            if VELOCITY_RESTRICT is False:
                vweight_update = tf.assign(vw,
                                       tf.add_n([nextvw, pdiffw, gdiffw]),
                                       validate_shape=True)
            else:
                vweight_update = tf.assign(vw,
                                       tf.maximum(
                                           tf.minimum(
                                               tf.add_n(
                                                   [nextvw, pdiffw, gdiffw]
                                               ),
                                               t_MVEL),
                                           -t_MVEL
                                       ),
                                       validate_shape=True)

            vweight_updates.append(vweight_update)
            vbias_update = None
            if VELOCITY_RESTRICT is False:
                vbias_update = tf.assign(vb,
                                     tf.add_n([nextvb, pdiffb, gdiffb]),
                                     validate_shape=True)
            else:
                vbias_update = tf.assign(vb,
                                     tf.minimum(
                                         tf.maximum(
                                             tf.add_n(
                                                 [nextvb, pdiffb, gdiffb]
                                             ),
                                             -t_MVEL),
                                         t_MVEL
                                     ),
                                     validate_shape=True)
            vbias_updates.append(vbias_update)
            weight_update = tf.assign(w, w + vw, validate_shape=True)
            weight_updates.append(weight_update)
            bias_update = tf.assign(b, b + vb, validate_shape=True)
            bias_updates.append(bias_update)


    loss = tf.reduce_mean(-tf.reduce_sum(label* tf.log(net), axis=0))
    if not LBPSO:
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_updates.append(accuracy)
    particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
    fit_update = tf.assign(pfit, particlebest, validate_shape=True)
    fit_updates.append(fit_update)
    if not LBPSO:
        globalbest = tf.cond(loss < gfit, lambda: loss, lambda: gfit)
        fit_update = tf.assign(gfit, globalbest, validate_shape=True)
        fit_updates.append(fit_update)
    pfitlist.append(pfit)
    control_update = tf.assign(t_MVEL, tf.multiply(t_MVEL, MAX_VEL_DECAY),validate_shape=True)
    control_updates.append(control_update)
    if not LBPSO:
       if HYBRID:
          optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
          hybrid_update = optimizer.minimize(loss)
          hybrid_updates.append(hybrid_update)
    if not LBPSO:
       assert len(weights) == len(biases)
       assert len(gweights) == len(gbiases)
       assert len(pweights) == len(pbiases)
       assert len(gweights) == len(weights)
       assert len(pweights) == len(weights)
    for i in range(len(weights)):
        
        # Particle Best
        pweight = tf.cond(loss < pfit, lambda: weights[i], lambda: pweights[i])
        fit_update = tf.assign(pweightslist[pno*len(weights)+i], pweight, validate_shape=True)
        fit_update = tf.assign(pweights[i], pweight, validate_shape=True)
        fit_updates.append(fit_update)
        pbias = tf.cond(loss < pfit, lambda: biases[i], lambda: pbiases[i])
        fit_update = tf.assign(pbiaseslist[pno*len(weights)+i], pbias, validate_shape=True)
        fit_update = tf.assign(pbiases[i], pbias, validate_shape=True)
        fit_updates.append(fit_update)
        if not LBPSO:
           
           # Global Best
           gweight = tf.cond(loss < gfit, lambda: weights[i], lambda: gweights[i])
           netlist[0]=net
           fit_update = tf.assign(gweights[i], gweight, validate_shape=True)

           fit_updates.append(fit_update)
           gbias = tf.cond(loss < gfit, lambda: biases[i], lambda: gbiases[i])
           fit_update = tf.assign(gbiases[i], gbias, validate_shape=True)
           fit_updates.append(fit_update)
    if not LBPSO:
        nets.append(net)
        losses.append(loss)
        print_prog_bar(iteration=pno + 1,
                   total=N_PARTICLES,
                   suffix=str_memusage('M'))
    len_weights=len(weights)

vweights=[]
vbiases=[]
if LBPSO:
    for pno in range(N_PARTICLES):
        weights = []
        biases = []
        pweights = []
        pbiases = []
        lbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=L_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'lbestrand',
        trainable=False)
        pfit = tf.Variable(math.inf, name='pno' + str(pno + 1) + 'fit')
        lfit = tf.Variable(math.inf, name='lbestfit')
        lfitlist.append(lfit)
        net = net_in
        pbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=P_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'pbestrand')

        # Append the random values so that the initializer can be called again
        random_values.append(pbestrand)
        random_values.append(lbestrand)

        # Define the parameters

        for idx, num_neuron in enumerate(LAYERS[1:]):

            net=netlist[pno*len_weights+idx]
            w=weightslist[pno*len_weights+idx]
            b=biaseslist[pno*len_weights+idx]
            pw=pweightslist[pno*len_weights+idx]
            pb=pbiaseslist[pno*len_weights+idx]
            vw=vweightslist[pno*len_weights+idx]
            vb=vbiaseslist[pno*len_weights+idx]

            # Multiply by the Velocity Decay
            nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
            nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

            # Differences between Particle Best & Current
            pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
            pdiffw = tf.multiply(pdiffw,c1)
       	    pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)
            pdiffb = tf.multiply(pdiffb,c2)

            l = (i-1)%N_PARTICLES
            r = (1+i)%N_PARTICLES
            nbestvalue=tf.cond(pfitlist[l]<pfitlist[r],lambda:pfitlist[l],lambda:pfitlist[r])
            pbestvalue=tf.cond(pfitlist[pno]<lfitlist[pno],lambda:pfitlist[pno],lambda:pfitlist[pno])
            lfitvalue=tf.cond(nbestvalue<pbestvalue,lambda:nbestvalue,lambda:pbestvalue)
            fitupdate=tf.assign(lfitlist[pno],lfitvalue)
            fit_updates.append(fitupdate)

            nbestweight=tf.cond(pfitlist[l]<pfitlist[r],lambda:pweightslist[l*len_weights+idx],lambda:pweightslist[r*len_weights+idx])
            pbestweight=tf.cond(pfitlist[pno]<lfitlist[pno],lambda:pweightslist[pno*len_weights+idx],lambda:lweightslist[pno*len_weights+idx])
            lweightvalue=tf.cond(nbestvalue<pbestvalue,lambda: nbestweight,lambda:pbestweight)
            fitupdate=tf.assign(lweightslist[pno*len_weights+idx],lweightvalue)
            fit_updates.append(fitupdate)

            nbestbias=tf.cond(pfitlist[l]<pfitlist[r],lambda:pbiaseslist[l*len_weights+idx],lambda:pbiaseslist[r*len_weights+idx])
            pbestbias=tf.cond(pfitlist[pno]<lfitlist[pno],lambda:pbiaseslist[pno*len_weights+idx],lambda:lbiaseslist[pno*len_weights+idx])
            lbiasvalue=tf.cond(nbestvalue<pbestvalue,lambda:nbestbias,lambda:pbestbias)
            fitupdate=tf.assign(lbiaseslist[pno*len_weights+idx],lbiasvalue)
            fit_updates.append(fitupdate)
            lw=lweightvalue
            lb=lbiasvalue
            if pno == 0:
                lweights.append(lw)
                lbiases.append(lb)

            # Differences between Local Best & Current
            ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
            ldiffw = tf.multiply(ldiffw,c1)
            ldiffb = tf.multiply(tf.subtract(lb, b), lbestrand)
            ldiffb = tf.multiply(ldiffb,c2)
            vweight_update = None
            if VELOCITY_RESTRICT is False:
                vweight_update = tf.assign(vw,
                                       tf.multiply(tf.add_n([nextvw, pdiffw, ldiffw]),chi),
                                       validate_shape=True)
            else:
                vweight_update = tf.assign(vw,
                                       tf.maximum(
                                           tf.minimum(
                                               tf.multiply(
                                               tf.add_n(
                                                   [nextvw, pdiffw, ldiffw]
                                               ),chi),
                                               t_MVEL),
                                           -t_MVEL
                                       ),
                                       validate_shape=True)

            vweight_updates.append(vweight_update)
            vbias_update = None
            if VELOCITY_RESTRICT is False:
                vbias_update = tf.assign(vb,
                                     tf.multiply(tf.add_n([nextvb, pdiffb, ldiffb]),chi),
                                     validate_shape=True)
            else:
                vbias_update = tf.assign(vb,
                                     tf.minimum(
                                         tf.maximum(
                                             tf.multiply(
                                             tf.add_n(
                                                 [nextvb, pdiffb, ldiffb]
                                             ),chi),
                                             -t_MVEL),
                                         t_MVEL
                                     ),
                                     validate_shape=True)
            vbias_updates.append(vbias_update)
            weight_update = tf.assign(w, w + vw, validate_shape=True)
            weight_updates.append(weight_update)
            bias_update = tf.assign(b, b + vb, validate_shape=True)
            bias_updates.append(bias_update)

        # Define loss for each of the particle nets
        loss = tf.nn.l2_loss(net - label)
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_updates.append(accuracy)
        particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
        localbest = tf.cond(loss < lfit, lambda: loss, lambda: lfit)
        fit_update = tf.assign(lfitlist[pno], localbest, validate_shape=True)
        fit_updates.append(fit_update)
        control_update = tf.assign(t_MVEL, tf.multiply(t_MVEL, MAX_VEL_DECAY),
                               validate_shape=True)
        control_updates.append(control_update)
        if HYBRID:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            hybrid_update = optimizer.minimize(loss)
            hybrid_updates.append(hybrid_update)

        # Particle Best
        for i in range(len(weights)):
            lweight = tf.cond(loss < lfit, lambda: weightslist[len_weights*pno+i], lambda: lweightslist[len_weights*pno+i])
            fit_update = tf.assign(lweightslist[len_weights*pno+i], lweight, validate_shape=True)
            fit_updates.append(fit_update)
            lbias = tf.cond(loss < lfit, lambda: biaseslist[len_weights*pno+i], lambda: lbiaseslist[len_weights*pno+i])
            fit_update = tf.assign(lbiaseslist[pno*len_weights+i], lbias, validate_shape=True)
            fit_updates.append(fit_update)

        # Update the lists
        nets.append(net)
        losses.append(loss)
        print_prog_bar(iteration=pno + 1,
                   total=N_PARTICLES,
                   suffix=str_memusage('M'))

msgtime('Completed\t\t:')

fcn_stats(LAYERS)
if LBPSO:
    for i in range(N_PARTICLES):
        gfit=tf.cond(gfit>pfitlist[i],lambda:pfitlist[i],lambda:gfit)



# Initialize the entire graph
init = tf.global_variables_initializer()
print('Graph Init Successful:')

# Define the updates which are to be done before each iterations
random_updates = [r.initializer for r in random_values]
updates = weightslist + biaseslist + \
    random_updates + vbiaseslist + vweightslist + \
    fit_updates + control_updates + hybrid_updates + pweightslist + pbiaseslist
req_list = losses, updates, gfit, gbiases, vweights, vbiases, gweights,accuracy_updates,nets
original_nets=nets
with tf.Session() as sess:
    sess.run(init)
    # Write The graph summary
    summary_writer = tf.summary.FileWriter('/tmp/logs', sess.graph_def)
    start_time = time.time()
    max_accuracy=0
    for i in range(N_ITERATIONS):
        # Reinitialize the Random Values at each iteration
        if HYBRID:
            sess.run(hybrid_update,feed_dict={net_in: X_train, label: y_train})
        _tuple = sess.run(req_list, feed_dict={
            net_in: X_train, label: y_train})
        _losses = None
        _losses, _, gfit, gbiases, vweights, vbiases, gweights, accuracy_updates, nets = _tuple
        if i==N_ITERATIONS-1:
            fh=open("gweights1.csv",'w')
            fh1=open("gbias1.csv",'w')
            for j in range(len(weights)):
                np.savetxt(fh,gweights[j])
                np.savetxt(fh1,gbiases[j])
        if (i + 1) % 1 == 0:
            print('Losses:', _losses, 'Iteration:', i+1)

            if max(accuracy_updates)>max_accuracy:
                max_accuracy=max(accuracy_updates)
            if not LBPSO:
                print('Gfit:', gfit)
                print('accuracy', max_accuracy)
            else:
                print('Best Particle', min(_losses))
                print('accuracy', max_accuracy)

    best_particle,_ = min(enumerate(_losses), key=operator.itemgetter(1))	
    #Testing
    for i in range(len(X_test)):
        print('Actual:', y_test[i], 'Predicted:', np.rint(sess.run(original_nets[best_particle], feed_dict={net_in: [X_test[i]]})))# XXX
    end_time = time.time()
    # Close the writer
    summary_writer.close()

    print('Total Time:', end_time - start_time)
