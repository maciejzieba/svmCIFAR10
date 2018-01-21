import sys
import numpy as np

from data_reader import load, get_data
from settings import DATA_DIR, LIBLINEAR_DIR
from scipy.stats import mode
from sklearn.metrics import accuracy_score


sys.path.append(LIBLINEAR_DIR)

from liblinearutil import *

data_trains = ['cifar_train_triplet_1024_x.npz',
               'cifar_train_triplet_100_x.npz',
               'cifar_train_triplet_2048_x.npz',
               'cifar_train_triplet_2048_L2_x.npz',
               'cifar_train_x.npz']
data_tests = ['cifar_test_triplet_1024_x.npz',
              'cifar_test_triplet_100_x.npz',
              'cifar_test_triplet_2048_x.npz',
              'cifar_test_triplet_2048_L2_x.npz',
              'cifar_test_x.npz']

cs = [0.015625, 0.031250, 0.031250, 0.250000, 0.031250]

_, trainy = load(DATA_DIR, subset='train')
_, testy = load(DATA_DIR, subset='test')

joined = []
for k in range(len(cs)):
    trainx = get_data(data_trains[k])
    testx = get_data(data_tests[k])
    trainx = trainx + np.random.normal(0, 0.3, trainx.shape)
    m = train(trainy, trainx, '-c '+str(cs[k]))
    p_label, p_acc, p_val = predict(testy, testx, m)
    joined.append(np.expand_dims(p_label, axis=0))
joined = np.transpose(np.concatenate(joined, axis=0),(1,0))
m_voting = []
for k in range(joined.shape[0]):
    m_voting.append(mode(joined[k])[0][0])

acc = accuracy_score(m_voting, testy)
print(acc)
