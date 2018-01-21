import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_reader import load, get_data
from settings import DATA_DIR, DATA_SEED

samples_per_class = 500

data_type = 'train' # 'test' or 'train'
data_name = 'cifar_train_triplet_100_x.npz'

datax = get_data(data_name)
_, datay = load(DATA_DIR, subset=data_type)
pca = PCA(n_components=2)
X_new = pca.fit_transform(datax)
print(datax.shape)
rng_data = np.random.RandomState(DATA_SEED)
inds = rng_data.permutation(X_new.shape[0])

X_new = X_new[inds]
datay = datay[inds]

plt.rcParams["figure.figsize"] = (15,12)
for j in range(10):
    txs = X_new[datay==j][:samples_per_class]
    plt.scatter(txs[:,0], txs[:,1])

plt.title('PCA 2D transform on ' + data_name, fontsize=20)
plt.xlabel('PC1', fontsize=18)
plt.ylabel('PC2', fontsize=18)
plt.savefig('pca_'+ data_name +'.png')
plt.show()