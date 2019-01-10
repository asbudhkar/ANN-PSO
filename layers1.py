import tensorflow as tf


# Activation Function
def activate(input_layer, act='relu', name='activation'):
    if act is None:
        return input_layer
    if act == 'relu':
        return tf.nn.relu(input_layer, name)
    if act == 'sqr':
        return tf.square(input_layer, name)
    if act == 'sqr_sigmoid':
        return tf.nn.sigmoid(tf.square(input_layer, name))
    if act == 'sigmoid':
        return tf.nn.sigmoid(input_layer, name)
    if act == 'softmax':
        return tf.nn.softmax(input_layer)


# Fully connected custom layer for PSO
# Supported activation function types : None,relu,sqr,sqr_sigmoid,sigmoid
def fc(input_tensor, n_output_units, scope, weight,bias,
       activation_fn='softmax', uniform=False):
    shape = [input_tensor.get_shape().as_list()[-1], n_output_units]
    # Use the Scope specified
    if n_output_units==3:
        activation_fn='softmax'
    else:
        activation_fn='sigmoid'
    with tf.variable_scope(scope):
        # Init Weights
        if uniform:
            weights = tf.Variable(weight)
        else:
            weights = tf.Variable(weight)
        # Init Biases
        biases = tf.Variable(bias)
        # Particle Best
        pbest_w = tf.Variable(weights.initialized_value(), name='pbest_w')
        pbest_b = tf.Variable(biases.initialized_value(), name='pbest_b')
        # Velocities
        vel_weights = tf.Variable(tf.random_uniform(
            shape=shape,
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_weights')
        vel_biases = tf.Variable(tf.random_uniform(
            shape=[n_output_units],
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_biases')
        print("BAISES",biases)
        # Perform actual feedforward
        act = tf.matmul(input_tensor, weights) #+ biases
        print("ACTi",act)
        pso_tupple = [weights, biases,
                      pbest_w, pbest_b,
                      vel_weights, vel_biases]
        # Activate And Return
        return activate(act, activation_fn), pso_tupple


# Magnitude Clipper
# Magmax can be either a Tensor or a Float
def maxclip(tensor, magmax):
    # assertion commented out to allow usage of both Tensor & Integer
    # assert magmax > 0, "magmax argument in maxclip must be positive"
    return tf.minimum(tf.maximum(tensor, -magmax), magmax)
