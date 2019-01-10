import itertools
from functools import reduce
import operator
import tensorflow as tf
import time
import random
import math
import argparse
import parseutils as pu
from layers1 import maxclip, fc
from utils import msgtime, str_memusage, print_prog_bar, fcn_stats, chical
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
tf.logging.set_verbosity(tf.logging.ERROR)

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

# Store model weights and biases
with open('gweights1.csv', 'rt') as f,open('gweight.csv', 'wt') as f_out:
     reader = csv.reader(f)
     writer = csv.writer(f_out)
     for row in reader:
        writer.writerow(row)

with open('gbias1.csv', 'rt') as f,open('gbias.csv', 'wt') as f_out:
     reader = csv.reader(f)
     writer = csv.writer(f_out)
     i=0
     for row in reader:    
        writer.writerow(row)

# Initialize placeholders
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

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
    # Test sample
    parser.add_argument('--test', nargs='+', type=pu.pfloat,
                        help='Test sample', default=[4.6,3.2,1.4,0.2 ])

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
PRINT_ITER = args.pi

#Test Sample
TEST = args.test

CHI = 1 

# Basic Neural Network Definition
# Simple feedforward Network
LAYERS = [N_IN] + HIDDEN_LAYERS + [3]
print('Network Structure\t:', LAYERS)


t_VELOCITY_DECAY = tf.constant(value=VELOCITY_DECAY,
                               dtype=tf.float32,
                               name='vel_decay')
t_MVEL = tf.Variable(MAX_VEL,
                     dtype=tf.float32,
                     name='vel_restrict',
                     trainable=False)

net_in = tf.placeholder(dtype=tf.float32,
                        shape=[None, 4],
                        name='net_in')

print('Mem Usage\t\t:', str_memusage(datatype='M'))
msgtime('Building Network\t:')

# MULTI-PARTICLE NEURAL NETS
losses = []
nets = []
pweights = []
pbiases = []
vweights = []
vbiases = []
random_values = []

# Positional Updates
bias_updates = []
weight_updates = []
bweight=[]

# Velocity Updates
vweight_updates = []
vbias_updates = []

# Fitness Updates
fit_updates = []
accuracy_updates=[]

# Control Updates - Controling PSO inside tf.Graph
control_updates = []

# Hybrid Updates - Using of PSO + Traditional Approaches
hybrid_updates = []

gweights = None
gbiases = None
gfit = None
bpno=tf.Variable(0,name='bpno')
if not LBPSO:
    gweights = []
    gbiases = []
    gfit = tf.Variable(math.inf, name='gbestfit', trainable=False)

fcn_stats(LAYERS)
for pno in range(N_PARTICLES):
    weights = []
    biases = []
    pweights = []
    pbiases = []
    lweights = None
    lbiases = None
    if LBPSO:
        # Initialize the list
        lweights = []
        lbiases = []
    pbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=P_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'pbestrand',
        trainable=False)
    gbestrand = None
    lbestrand = None
    if not LBPSO:
        gbestrand = tf.Variable(tf.random_uniform(
            shape=[], maxval=G_BEST_FACTOR),
            name='pno' + str(pno + 1) + 'gbestrand',
            trainable=False)
    else:
        lbestrand = tf.Variable(tf.random_uniform(
            shape=[], maxval=L_BEST_FACTOR),
            name='pno' + str(pno + 1) + 'lbestrand',
            trainable=False)

    # Append the random values so that the initializer can be called again
    random_values.append(pbestrand)
    if not LBPSO:
        random_values.append(gbestrand)
    else:
        random_values.append(lbestrand)
    pfit = None
    with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
        init = tf.constant(math.inf)
        pfit = tf.get_variable(name=str(pno + 1),
                               initializer=init)

    pfit = tf.Variable(math.inf, name='pno' + str(pno + 1) + 'fit')

    localfit = None
    if LBPSO:
        localfit = tf.Variable(math.inf, name='pno' + str(pno + 1) + 'lfit')
    net = net_in
    # Define the parameters

    for idx, num_neuron in enumerate(LAYERS[1:]):
        print("ID",idx)
        print("NET",net)
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        ifile = open('gweight.csv', "r")
        print("N",num_neuron)
        
        with open("gweight.csv") as ifile:
             weights = [next(ifile) for x in range(net.get_shape().as_list()[-1])]
             weights = list(map(lambda s: s.strip(), weights))
             weights = [[(float(j)) for j in i.split()] for i in weights]
       
        with open("gbias.csv") as ifile:
            biases = [next(ifile) for x in range(num_neuron)]
            biases = list(map(lambda s: s.strip(), biases))
            biases = [[(float(j)) for j in i.split()] for i in biases]
        
        lines = open('gweight.csv').readlines()
        open('gweight.csv', 'w').writelines(lines[net.get_shape().as_list()[-1]:])
        lines = open('gbias.csv').readlines()
        open('gbias.csv', 'w').writelines(lines[num_neuron:])
        
        net, pso_tupple = fc(input_tensor=net,
                             n_output_units=num_neuron,
                             activation_fn='sigmoid',
                             scope=layer_scope,
                             uniform=False,weight=weights,bias=biases)
        w, b, pw, pb, vw, vb = pso_tupple
        
        print("NET",net)

    if HYBRID:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        hybrid_update = optimizer.minimize(loss)
        hybrid_updates.append(hybrid_update)

    print_prog_bar(iteration=pno + 1,
                   total=N_PARTICLES,
                   suffix=str_memusage('M'))


msgtime('Completed\t\t:')

# Initialize the entire graph
init = tf.global_variables_initializer()
msgtime('Graph Init Successful\t:')

# Define the updates which are to be done before each iterations
random_updates = [r.initializer for r in random_values]
updates = weight_updates + bias_updates + \
    random_updates + vbias_updates + vweight_updates + \
    fit_updates + control_updates + hybrid_updates
req_list = None
if not LBPSO:
    req_list = losses, accuracy_updates
else:
    req_list = losses, accuracy_updates
with tf.Session() as sess:
    sess.run(init)

    # Write The graph summary
    summary_writer = tf.summary.FileWriter('/tmp/tf/logs', sess.graph_def)
    start_time = time.time()
    max_accuracy=0
    for i in range(N_ITERATIONS):
        if HYBRID:
            sess.run(hybrid_update,feed_dict={X_data: X_train, y_target: y_train})
    for i in range(len(X_test)):
        predicted=sess.run(net, feed_dict={net_in: [X_test[i]]})
        predicted=flat_list = [item for sublist in predicted for item in sublist]
        predicted=np.rint(predicted)
        print('Actual:', y_test[i], 'Predicted:', np.rint(predicted))
   
   
    predicted=sess.run(net, feed_dict={net_in: [TEST]})
    predicted=flat_list = [item for sublist in predicted for item in sublist]
    predicted=np.rint(predicted)
    s=np.asarray([1,0,0])
    ve=np.asarray([0,1,0])
    vi=np.asarray([0,0,1])  # Get the predicted class (index)
    fh=open('output1.txt','w')
    if(np.array_equal(s,predicted)):
        print("  I think:",TEST," is Iris Setosa")
        fh.write("Setosa")
    elif(np.array_equal(ve,predicted)):
        fh.write("Versicolor")
        print("  I thinx		k:",TEST," is Iris Versicolor")
    else:
        fh.write("Virginica")
        print("  I think:",TEST," is Iris Virginica")
    summary_writer.close()
    end_time = time.time()
    print('Total Time:', end_time - start_time)
