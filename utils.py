import numpy as np
import matplotlib.pyplot as plt


def save_imgs(imgs):
    plt.figure(figsize=(8, 8))
    imgs = imgs.numpy()
    fig = plt.imshow(np.transpose(imgs, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('examples.png')
