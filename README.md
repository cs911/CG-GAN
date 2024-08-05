This is the code for the paper "A Novel Confidence Guided Training Method for Conditional
GANs with Auxiliary Classifier".
The code is modified from [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).

## Requirements

First, install PyTorch meeting your environment (at least 1.7, recommmended 1.10):
```bash
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Then, use the following command to install the rest of the libraries:
```bash
pip3 install tqdm ninja h5py kornia matplotlib pandas scikit-learn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg
```

For installing all the requirements use the following command:

```
conda env create -f environment.yml -n base
```

Before starting, users should login wandb using their personal API key.

    wandb login PERSONAL_API_KEY

# Dataset

* CIFAR10/CIFAR100: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny ImageNet, ImageNet, or a custom dataset:
  1. download [Tiny ImageNet](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4), [Baby ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [Papa ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [Grandpa ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [ImageNet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
data
└── ImageNet, Tiny_ImageNet, Baby ImageNet, Papa ImageNet, or Grandpa ImageNet
    ├── train
    │   ├── cls0
    │   │   ├── train0.png
    │   │   ├── train1.png
    │   │   └── ...
    │   ├── cls1
    │   └── ...
    └── valid
        ├── cls0
        │   ├── valid0.png
        │   ├── valid1.png
        │   └── ...
        ├── cls1
        └── ...
```

When training and evaluating, we used the command below.
```
"nkl" in "ACGAN-Mod-Big-nkl.yaml" denotes our method rCG-GAN

"lab" in "ACGAN-Mod-Big-lab.yaml" denotes our method fCG-GAN
```

--------For CIFAR10/CIFAR100:
```
CUDA_VISIBLE_DEVICES=1   python3 code/main.py -t -hdf5 -l -batch_stat  -metrics is fid prdc -ref "test" -cfg ./code/configs/CIFAR100/ACGAN-Mod-Big-nkl.yaml -data cifar100 -save save 
```
--------For Baby/Papa/Grandpa-ImageNet and Tiny-ImageNet:
```
CUDA_VISIBLE_DEVICES=1  python3 code/main.py -t -hdf5 -l -batch_stat  -metrics is fid prdc -ref "valid" -cfg ./code/configs/Papa_ImageNet/ACGAN-Mod-Big-nkl.yaml -data Papa_ImageNet -save save 
```
--------For ImageNet
```
CUDA_VISIBLE_DEVICES=1  python3 code/main.py -t -hdf5 -l -sync_bn   -metrics is fid prdc -ref "valid" -cfg ./code/configs/ImageNet/ACGAN-Mod-Big-nkl.yaml -std_stat -std_max 256 -std_step 256 -mpc -data ImageNet -save save 
```

