import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Plot image examples.
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img

def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0, stretch=False):

    if stretch:
        imgs = img_stretch(imgs)
    imgs = np.array(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i*grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img



