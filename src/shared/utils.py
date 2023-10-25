import math
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from IPython.display import display, clear_output


def to_tensor(data, device='cpu', dtype=torch.float32, requires_grad=False):
    return torch.tensor(
        data,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def to_numpy(tensor):
    return tensor.cpu().data.numpy()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0

    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass

    experiment_id += 1
    parent_dir = f'{parent_dir}/run{experiment_id}'
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def plot_rays(rays,
              step=None, fig=None, ax=None,
              title=''):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    sample = rays[::step]
    X = sample[..., 0].cpu().numpy()
    Y = sample[..., 1].cpu().numpy()
    Z = sample[..., 2].cpu().numpy()

    ax.cla()
    ax.quiver(0, 0, 0, X, Y, Z, label='directions')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    display(fig)
    clear_output(wait=True)

    return fig, ax


def show_img(img_batch,
             fig=None, ax=None, cmap='gray',
             title='', write_path=None, stdout=False,
             three_d=False):
    nrows = math.ceil(img_batch.shape[0] / 2)
    ncols = min(2, img_batch.shape[0])

    if fig is None:
        subplot_kw = dict(projection='3d') if three_d else None
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, subplot_kw=subplot_kw)

    for i in range(nrows):
        for j in range(ncols):
            idx = i*ncols + j
            ax[i, j].cla()
            ax[i, j].axis('off')
            if idx < img_batch.shape[0]:
                img = img_batch[idx]
                if img.shape[-1] == 2:
                    ax[i, j].axis('on')
                    x = img[..., 0]
                    y = img[..., 1]
                    ax[i, j].scatter(y, x, marker='.')
                    ax[i, j].set_xlim(0, img.shape[1])
                    ax[i, j].set_ylim(img.shape[0], 0)
                    ax[i, j].xaxis.tick_top()
                    ax[i, j].yaxis.tick_left()
                    ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax[i, j].yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax[i, j].set_aspect('equal', adjustable='box')
                    ax[i, j].grid(True)
                else:
                    if three_d:
                        _x = np.arange(img.shape[0])
                        _y = np.arange(img.shape[1])
                        _xx, _yy = np.meshgrid(_x, _y)
                        x, y = _xx.ravel(), _yy.ravel()
                        z = np.zeros_like(x)
                        dz = img.T.ravel()
                        norm = colors.Normalize(dz.min(), dz.max())
                        color = plt.get_cmap(cmap)(norm(dz))
                        ax[i, j].bar3d(x, y, z, 1, 1, dz, color=color)
                    else:
                        ax[i, j].imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax[i, j].imshow(np.ones(img_batch.shape[1:]))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if stdout:
        display(fig)
        clear_output(wait=True)

    if write_path is not None:
        plt.savefig(f'{write_path}/{title}.png', bbox_inches='tight')

    return fig, ax
