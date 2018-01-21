from data_reader import load
from settings import DATA_DIR
import numpy as np

trainx, trainy = load(DATA_DIR, subset='train')
print('The training data is composed of: ' + str(np.shape(trainx)[0]) + ' examples.')
testx, testy = load(DATA_DIR, subset='test')
print('The testing data is composed of: ' + str(np.shape(testx)[0]) + ' examples.')