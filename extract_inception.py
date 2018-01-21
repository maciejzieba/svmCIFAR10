import pickle
import lasagne
import numpy as np
import theano as th
import theano.tensor as T
import lasagne.layers as ll

from data_reader import load
from settings import DATA_DIR

from inception_v3 import build_network, preprocess


def extract(data, layer, batch_size):
    nr_batches_train = int(data.shape[0]/batch_size)
    x_temp = T.tensor4()
    features = ll.get_output(layer, x_temp , deterministic=True)
    extract_features = th.function(inputs=[x_temp ], outputs=features)
    output_features = []
    for t in range(nr_batches_train):
        train_temp = data[t*batch_size:(t+1)*batch_size]
        tx_resized = []
        for n in range(batch_size):
            tx_resized.append(preprocess(np.transpose(train_temp[n],(1,2,0))))
        tx_resized = np.concatenate(tx_resized, axis=0)
        output_features.append(extract_features(tx_resized))

    return np.concatenate(output_features, axis=0)

with open('inception_v3.pkl', 'rb') as f:
    params = pickle.load(f)

net = build_network()
lasagne.layers.set_all_param_values(net['softmax'], params['param values'])

trainx, _ = load(DATA_DIR, subset='train')
testx, _ = load(DATA_DIR, subset='test')

minibatch_size = 10
feature_layer = net['pool3']
print("Extracting features from train data...")
train_features = extract(trainx, feature_layer, minibatch_size)
print("Extracting features from test data...")
test_features = extract(testx, feature_layer, minibatch_size)

print(train_features.shape)
print(test_features.shape)

np.savez_compressed('cifar_train_x', train_features)
np.savez_compressed('cifar_test_x', test_features)