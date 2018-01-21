import sys
import numpy as np
from scipy.stats import mode

from sklearn.metrics import accuracy_score

from data_reader import load, get_data
from settings import DATA_DIR, LIBLINEAR_DIR

sys.path.append(LIBLINEAR_DIR)

from liblinearutil import *

data_train = get_data('cifar_train_triplet_100_x.npz')
data_test = get_data('cifar_test_triplet_100_x.npz')

# For more general
K = 10
cs =  0.250000

_, trainy = load(DATA_DIR, subset='train')
_, testy = load(DATA_DIR, subset='test')

joined = []
for k in range(K):
    ind1 = np.random.choice(data_train.shape[0], data_train.shape[0])
    trainx_temp = data_train[ind1]
    trainx_temp = trainx_temp + np.random.normal(0, 0.3, trainx_temp.shape)
    trainy_temp = trainy[ind1]
    m = train(trainy_temp, trainx_temp, '-c '+str(cs))
    p_label, p_acc, p_val = predict(testy, data_test, m)
    joined.append(np.expand_dims(p_label, axis=0))
joined = np.transpose(np.concatenate(joined, axis=0),(1,0))
m_voting = []
for k in range(joined.shape[0]):
    m_voting.append(mode(joined[k])[0][0])

acc = accuracy_score(m_voting, testy)
print(acc)
