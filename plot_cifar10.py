import plotting
import numpy as np
from data_reader import load
from settings import DATA_DIR, DATA_SEED

trainx, trainy = load(DATA_DIR, subset='train')
rng_data = np.random.RandomState(DATA_SEED)
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
for j in range(10):
    txs.append(trainx[trainy==j][:10])

txs = np.concatenate(txs, axis=0)

img_bhwc = np.transpose(txs, (0, 2, 3, 1))
img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title='CIFAR10 samples')
plotting.plt.savefig("cifar_sample.png")