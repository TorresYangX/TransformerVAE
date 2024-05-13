# Reconstructing Trajectories with Transformer Variational AutoEncoder

## TimeLine
- Dataset process: Geolife and Porto (done)
- NVAE model (done)
- baseline model:
    - EDR (done)
    - EDwP (done)
    - t2vec (done)
    - TrjSR
    - E2DTC
    - TrajCL
- metric:
    - cityemd (done)
    - NMD (done)
    - NMA (done)
    - RRNSA (done)


## Installation

- Python 3.9

- ```bash
  conda env create -f environment.yml   # Create environment
  conda activate nvib                   # Activate environment
  ```

  *NB* This environment also installs the nvib package which is assumed to be in a directory one level above the current directory.

- Dataset can be downloaded from: [Geolife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) and [Porto](https://tianchi.aliyun.com/dataset/94216).

## Quick Start

To train NVAE or baseline models,  and encode latent vectors:

```bash
python Train.py --dataset porto --model model_name
```

To retrieve reconstructed trajectories:

```bash
python retrieve.py --dataset porto --model model_name
```

To test reconstructed trajectories with cityemd or Mobility Tableau metrics:

```bash
python cityemd.py --dataset porto --model model_name
python evaluate_yao.py --dataset porto --model model_name
```

## Datasets

To use your own datasets, you may need to create your own pre-processing script like `./utils/preprocessing_porto.py`. Also, the configuration is required to fill into `dataset_config.py`. (See `./utils/preprocessing_porto.py` and `dataset_config.py` for more details.)

## Baselines

To use your own baselines, you may need to create your own baseline script like `./baseline/AE.py`. Also, the configuration is required to fill into `model_config.py`. (See `./baseline/AE.py` and `model_config.py` for more details.)

## Repository Structure

```
├── baseline
    ├── AE.py
    ├── t2vec.py
    ├── transformer.py
    └── VAE.py
├── data
    └── .gitkeep
├── exp
    └── .gitkeep
├── model
    ├── NVAE.py
    ├── NVAE_trainer.py
├── utils
    ├── dataloader.py
    ├── preprocessing_geolife.py
    ├── preprocessing_porto.py
    ├── seq2multi_row.py
    └── tool_funcs.py
├── .gitignore
├── cityemd.py
├── dataset_config.py
├── environment.yml
├── evaluate_yao.py
├── model_config.py
├── README.md
├── retrieve.py
└── Train.py

```