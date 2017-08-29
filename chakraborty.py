
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


#takeInstances2onevar(training_set, idpred, ninputneurons):
# a fazer


def takeInstances6(dset, nt, idpred):
    nLines = dset.shape[0]
    nCols = dset.shape[1]
    nt = 2 #numero de instantes de tempo anteriores
    x_ = np.zeros([nLines-(nt+1), nCols*nt], dtype='float32')
    y_ = np.zeros([nLines - (nt + 1),1], dtype='float32')

    for k in range(nLines-(nt+1)):
        feat = dset[k:k+nt,:]
        x_[k,:] = feat.flatten()
        y_[k] = dset[k+nt,idpred]
    return x_, y_


def take8Previous(data, l,c):
    x_ = np.zeros([1, 8], dtype='float32')
    k = 0
    for i in reversed(range(l)):
        for j in reversed(range(c)):
            if k < 8:
                x_[k] = data[i,j]
            else:
                break
            k = k + 1
    return x_



def takeInstances8(data):
    nLines = data.shape[0]

    x_ = np.zeros([data.shape[0]*data.shape[1]-9, 8], dtype='float32')
    y_ = np.zeros([data.shape[0]*data.shape[1]-9,1], dtype='float32')

    x_[0,:] = take8Previous(data,2,2)
    y_[0] = data[2,2]
    k=1
    for i in range(2,nLines):
        for j in range(3,data.shape[1]):
            x_[k,:] = np.concatenate((x_[k-1,1:], y_[k-1]), axis=0)
            y_[k] = data[i,j]
    return x_, y_


def normalizeFeaturesSigmoid(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = 1.0 / (1.0 + np.exp(-data[i,j]))
    return data


def denormalizeFeatureSigmoid(f):
    return (np.log(f) - np.log(1.0 -f))


def normalizeFeaturesMax(data):
    max_atr = data.max(axis=0)
    data = data / max_atr
    return data, max_atr


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_h = tf.add(tf.matmul(x, weights['h']), biases['b'])
    layer_h = tf.nn.sigmoid(layer_h)
    out_layer = tf.matmul(layer_h, weights['out']) + biases['out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def shuffle_samples(x,y):
    n = y.shape[0]
    v = np.arange(n)
    np.random.shuffle(v)
    x_= [x[i,:] for i in v]
    y_ = [y[i, :] for i in v]
    return np.array(x_), np.array(y_)


def takeInstances(data, idpred, ninputneurons):
    if ninputneurons==6:
        x_, y_ = takeInstances6(data, idpred)
    elif ninputneurons==8:
        x_, y_ = takeInstances8(data)
    # elif ninputneurons==2:
    #     x_train, y_train = takeInstances2onevar(training_set, idpred, ninputneurons)
    return x_, y_

def neuralnetwork(data, ninputneurons, nhiddenneurons, nepochs, hop):

    nt = 2
    idpred = 2


    # Parametros do algoritmo de aprendizado
    learning_rate = 0.003
    momentum = 0.006
    training_epochs = nepochs

    display_step = 100
    # tf.set_random_seed(1234)

    # Parametros da arquitetura da rede
    n_hidden = nhiddenneurons  # hidden layer neurons
    n_input = ninputneurons  # data input (x_t, y_t, z_t, x_t-1, y_t-1, z_t-1)
    n_out = 1  # output neuron

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=[None, n_input])
    y = tf.placeholder(tf.float32, shape=[None, n_out])

    # Store layers weight & bias
    weights = {
        'h': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.1))
    }
    biases = {
        'b': tf.Variable(tf.truncated_normal([n_hidden], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_out], stddev=0.1))
    }

    # Construct model
    out = multilayer_perceptron(x, weights, biases)

    # Define mse and optimizer
    mse = tf.sqrt(tf.reduce_mean(tf.square(out - y)))
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        init.run()

        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            x_train_, y_train_ = shuffle_samples(x_train, y_train)

            sess.run(training_op, feed_dict={x: x_train_, y: y_train_})
            # Display logs per epoch step
            if epoch % display_step == 0:
                mserror = mse.eval(feed_dict={x: x_train, y: y_train})
                print(epoch, "\tMSE", mserror)
        print("Optimization Finished!")

        # Test model
        y_pred = sess.run(out, feed_dict={x: x_test})
        mserror = mse.eval(feed_dict={x: x_test, y: y_test})
        print("MSE in test: ", mserror)

        for i in range(y_test.shape[0]):
            print('Previsao: ', denormalizeFeatureSigmoid(y_pred[i]), '\tReal:', denormalizeFeature(y_test[i]))
            # print('Previsao: ', y_pred[i], '\tReal:', y_test[i])

def takeparameters(argv):
    data = np.loadtxt(argv[1],
                      dtype='f', delimiter=',')
    normalization = argv[3]
    nneuronsinput = argv[5]
    nneuronshidden = argv[7]
    learningrate = argv[9]
    nepochs = argv[11]
    predictionmode = argv[13]
    return data, normalization, nneuronsinput, nneuronshidden, \
    learningrate, nepochs,predictionmode



if __name__ == "__main__":

    data, normalization, nneuronsinput, nneuronshidden, \
    learningrate, nepochs,predictionmode = takeparameters(sys.argv[1:])

    dataf = data[:, 1:] # excluindo a primeira columa a qual o instante de tempo

    idpred = 2 # atributo para predição
    ntrainingperiods = 90

    if normalization=='sigmoid':
        dataf = normalizeFeaturesSigmoid(dataf)
    elif normalization=='max':
        dataf, maxatr = normalizeFeaturesMax(dataf)

    N = data.shape[0]  # numero de coletas
    col = data.shape[1]  # col-1 = numero de atributos

    training_set = dataf[0:ntrainingperiods, :]
    test_set = dataf[ntrainingperiods:N, :]

    x_train, y_train = takeInstances(training_set, idpred, nneuronsinput)

    x_test, y_test = takeInstances(test_set, idpred, nneuronsinput)





