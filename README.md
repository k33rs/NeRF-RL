# NeRF-RL

## Install nerfstudio (Windows) [OUTDATED]
Install Visual Studio 2022 (including `Desktop Development with C++` and `Windows Universal CRT SDK`).

Install cuda toolkit 1.8 and reboot system.

Open anaconda prompt:

    > conda create -n nerfstudio -y python=3.8
    > conda activate nerfstudio
    > python -m pip install --upgrade pip
    > pip install ninja
    > pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    > conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

In Visual Studio 2022, open folder `\miniconda3\envs\nerfstudio\Lib\site-packages` and open the terminal (solution found [here](https://github.com/nerfstudio-project/nerfstudio/issues/1177#issuecomment-1418298254)):

    > git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
    > cd tiny-cuda-nn
    > cmake . -B build
    > cmake --build build --config RelWithDebInfo -j

Back to anaconda prompt:

    > pip install nerfstudio==0.3.4 git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

## Install project dependencies

    > pip install -r requirements.txt

## Useful nerfstudio commands

    > ns-process-data images --data <train set dir> --eval-data <val set dir> --output-dir <data dir>
    > ns-train nerfacto --data <data dir> --output-dir <output dir> --load-checkpoint <ckpt file>
    > ns-viewer --load-config <yml file>
    > ns-export pointcloud --load-config <yml file> --output-dir <output dir> --normal-method open3d --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1