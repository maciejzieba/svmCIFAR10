import numpy as np
from skimage.feature import hog
from skimage import color

from data_reader import load
from settings import DATA_DIR


def extract_hog_features(data, orientations=16, pixels_per_cell=(8, 8),cells_per_block=(2, 2)):
    data = np.transpose(data,(0,2,3,1))
    features = []
    for i in range(np.shape(data)[0]):
        print(i)
        image = color.rgb2gray(data[i])
        fd = np.expand_dims(hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,transform_sqrt=True),axis=0)
        features.append(fd)
    return np.concatenate(features,axis=0)

trainx, _ = load(DATA_DIR, subset='train')
testx, _ = load(DATA_DIR, subset='test')

hog_features_train = extract_hog_features(trainx)
hog_features_test = extract_hog_features(testx)
np.savez_compressed('cifar_train_hog_x', hog_features_train)
np.savez_compressed('cifar_test_hog_x', hog_features_test)