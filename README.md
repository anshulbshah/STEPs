# STEPs : Self-Supervised Key Step Extraction from Unlabeled Procedural Videos

- Installing the environment : 
    - `conda create -n steps python=3`
    - `conda activate steps`
    - Install packages: 
        - Install PyTorch from [here](https://pytorch.org/get-started/locally/)
        - Install additional packages using : `pip install tqdm pandas scikit-learn==0.24.2`

- Download data
    - Download features for RGB-ResNet50 and OF-RAFT from [this](https://1drv.ms/u/s!AnyfQOW-gC9CbzhVoC9Hx96rRTw?e=BLZviD) location : Size : ~2.6G
    - Place them in `$ROOT/Data/Meccano`
    - Download annotations for Meccano dataset from [EgoProceL](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/EgoProceL-download-README.md) and place them in `$ROOT/Data/Meccano/annotations/`

- Train a model
    - Run `python main.py --train --random_seed $seed` to train a model. Results in the paper are reported as average over three seeds.
    - Trained model can be evaluated for Key Step Localization by running `python main.py --test --load_checkpoint saved_models/iccv2023_STEPs_Meccano/300.pth`

- Acknowledgements
    - We use annotations for Meccano and Evaluation code from EgoProceL (Bansal et al.)