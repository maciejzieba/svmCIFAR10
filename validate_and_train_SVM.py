import sys
import numpy as np
from settings import LIBLINEAR_DIR
from data_reader import load, get_data
from settings import DATA_DIR

sys.path.append(LIBLINEAR_DIR)

from liblinearutil import *

train_set = 'cifar_train_triplet_100_x.npz'
test_set = 'cifar_test_triplet_100_x.npz'

# Perform only model selection (finding best C for linear SVM using CV)
only_model_selection = False

# Save final model
save_model = False
model_name = 'model_best_triplet'

trainx = get_data(train_set)
_, trainy = load(DATA_DIR, subset='train')

testx = get_data(test_set)
_, testy = load(DATA_DIR, subset='test')

result = train(trainy, trainx, '-C')
if not only_model_selection:
    m = train(trainy, trainx, '-c '+str(result[0]))
    p_label, p_acc, p_val = predict(testy, testx, m)
    if save_model:
        save_model(model_name, m)