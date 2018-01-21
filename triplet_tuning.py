import lasagne.layers as LL
import sys

import time

import nn
from data_reader import get_data, load
import theano.tensor as T
import theano as th
import numpy as np
from settings import DATA_DIR

batch_size = 100
learning_rate = 0.0003
seed = 1
n_epochs = 200

save_model_as = 'triplet_extractor.npz'
#setting = [4048, 4048, 1024]
#setting = [2048, 1048, 100]
setting = [4048, 4048, 2048]

''' '' if we use loss from https://arxiv.org/abs/1704.02227
'L2' if we use loss max(d_+ - d_- + \lambda, 0), where \lambda=10.0'''
l_type = 'L2'

layers = [LL.InputLayer(shape=(None, 2048))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.DenseLayer(layers[-1], num_units=setting[0]))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=setting[1]))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=setting[2]))

trainx = get_data('cifar_train_x.npz')
_, trainy = load(DATA_DIR, subset='train')

print(trainx.shape)

x_lab = T.matrix()
output_lab = LL.get_output(layers[-1], x_lab, deterministic=False)

def get_triplets(prediction,size):
    a = prediction[0:size] # query case (positive)
    b = prediction[size:2*size] # positive case
    c = prediction[2*size:3*size] # negative

    return a,b,c

a_lab,b_lab,c_lab = get_triplets(output_lab,batch_size)


def loss_labeled(a,b,c):
    n_plus = T.sqrt(T.sum((a - b)**2, axis=1));
    n_minus = T.sqrt(T.sum((a - c)**2, axis=1));
    z = T.concatenate([n_minus.dimshuffle(0,'x'),n_plus.dimshuffle(0,'x')],axis=1)
    z = nn.log_sum_exp(z,axis=1)
    return n_plus,n_minus,z

if l_type == 'L2':
    n_plus = T.sum((a_lab - b_lab)**2, axis=1)
    n_minus = T.sum((a_lab - c_lab)**2, axis=1)
    dist = n_plus - n_minus + 10.0
    loss_lab = T.mean(dist*T.gt(dist,0.0))
else:
    n_plus_lab,n_minus_lab,z_lab = loss_labeled(a_lab,b_lab,c_lab)
    loss_lab = -T.mean(n_minus_lab) + T.mean(z_lab)

lr = T.scalar()
disc_params = LL.get_all_params(layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]

train_batch_disc = th.function(inputs=[x_lab, lr], outputs=loss_lab, updates=disc_param_updates+disc_avg_updates)


nr_batches_train = int(trainx.shape[0]/batch_size)

rng = np.random.RandomState(seed)

for epoch in range(n_epochs):
    begin = time.time()
    trainx_temp = trainx.copy()
    trainy_temp = trainy.copy()
    perm = rng.permutation(trainx_temp.shape[0])
    trainx_temp = trainx_temp[perm]
    trainy_temp = trainy_temp[perm]
    Pos = trainx_temp.copy()
    Neg = trainx_temp.copy()
    perm_pos = rng.permutation(int(trainx_temp.shape[0]/10))
    perm_neg = rng.permutation(int(9*trainx_temp.shape[0]/10))
    for k in range(10):
        Pos[trainy_temp==k] = trainx_temp[trainy_temp==k][perm_pos]
        Neg[trainy_temp==k] = trainx_temp[trainy_temp!=k][perm_neg[:np.shape(perm_pos)[0]]]
    lr = np.cast[th.config.floatX](learning_rate * np.minimum(3. - epoch/400., 1.))
    loss_lab = 0.
    for t in range(nr_batches_train):
        #print(t)
        temp = trainx_temp[t*batch_size:(t+1)*batch_size]
        temp = np.concatenate((temp, Pos[t*batch_size:(t+1)*batch_size]),axis=0)
        temp = np.concatenate((temp, Neg[t*batch_size:(t+1)*batch_size]),axis=0)
        ll= train_batch_disc(temp,lr)
        loss_lab += ll
    loss_lab /= nr_batches_train

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f" % (epoch, time.time()-begin, loss_lab ))
    sys.stdout.flush()
np.savez(save_model_as, *[p.get_value() for p in disc_params])
x_temp = T.matrix()
features = LL.get_output(layers[-1], x_temp, deterministic=True)
extract_features = th.function(inputs=[x_temp ], outputs=features)

train_features = []
for t in range(nr_batches_train):
    train_features.append(extract_features(trainx[t*batch_size:(t+1)*batch_size]))

train_features = np.concatenate(train_features,axis=0)
print(train_features.shape)

testx = get_data('cifar_test_x.npz')

nr_batches_test = int(testx.shape[0]/batch_size)

test_features = []
for t in range(nr_batches_test):
    test_features.append(extract_features(testx[t*batch_size:(t+1)*batch_size]))

test_features = np.concatenate(test_features,axis=0)
print(test_features.shape)

if l_type == 'L2':
    np.savez_compressed('cifar_train_triplet_+'+str(setting[-1])+'_L2_x', train_features)
    np.savez_compressed('cifar_test_triplet_+'+str(setting[-1])+'_L2_x', test_features)
else:
    np.savez_compressed('cifar_train_triplet_+'+str(setting[-1])+'_x', train_features)
    np.savez_compressed('cifar_test_triplet_+'+str(setting[-1])+'_x', test_features)