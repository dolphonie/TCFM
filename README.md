# Trajectory Forecasting and Generation with Conditional Flow Matching &nbsp;&nbsp;

## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

## Usage

Train a model with:
```
python scripts/train.py --config config.maze2d 
```

## Building your Own Dataset

This README explains how to structure your dataset for use with the `GeneralTrajectoryDataset` class. The dataset is organized hierarchically, with individual trajectory data stored in separate folders.

### Folder Structure

```
root_folder/
│
├── trajectory_1/
│   ├── timestamp.npz
│   ├── longitude.npz
│   ├── latitude.npz
│   └── altitude.npz
│
├── trajectory_2/
│   ├── timestamp.npz
│   ├── longitude.npz
│   ├── latitude.npz
│   └── altitude.npz
│
├── trajectory_3/
│   ├── timestamp.npz
│   ├── longitude.npz
│   ├── latitude.npz
│   └── altitude.npz
│
└── ... (more trajectory folders)
```

### Explanation

1. **Root Folder**: This is the main directory containing all trajectory data. You'll provide the path to this folder when initializing the `GeneralTrajectoryDataset`.

2. **Trajectory Folders**: Each subfolder represents a single trajectory. Name these folders uniquely (e.g., by date, flight number, or any other identifier).

3. **Feature Files**: Within each trajectory folder, store individual feature data in separate `.npz` files. The default features are:
   - `timestamp.npz`
   - `longitude.npz`
   - `latitude.npz`
   - `altitude.npz`

   Each `.npz` file should contain a single array named 'arr_0'.

### File Format

- Use NumPy's `.npz` format to store the data.
- Each `.npz` file should contain a single array named 'arr_0'.
- Ensure all feature arrays within a trajectory have the same length.


### Config File

Once your dataset is added, go to the config file to change which features to predict and which features to condition the track generation on. 

```
'history_length': 10, 
'features_to_load': ['longitude', 'latitude', 'altitude'],
'packed_features': ['longitude', 'latitude', 'altitude'],
'normalization': {
    'longitude': {'min': -76.07, 'max': -73.91},
    'latitude': {'min': 40.28, 'max': 41.55},
    'altitude': {'min': 0, 'max': 20000},
},
```

- history_length: How many historicarl points to include
- features_to_load: Which features to predict through the model
- paocked_features: Which features to condition the generative process with
- normalization: Dictionary containing min and max values such that the data is distributed between -1 and 1 


## Docker

1. Build the container:
```
docker build -f azure/Dockerfile . -t diffuser
```

2. Test the container:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --logbase logs/docker"
```


## Running on Azure

#### Setup

1. Launching jobs on Azure requires one more python dependency:
```
pip install git+https://github.com/JannerM/doodad.git@janner
```

2. Tag the image built in [the previous section](#Docker) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`


## Reference
```
@misc{ye2024tcfm,
      title={Efficient Trajectory Forecasting and Generation with Conditional Flow Matching}, 
      author={Sean Ye and Matthew Gombolay},
      year={2024},
      eprint={2403.10809},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2403.10809}, 
}
```


## Acknowledgements

The repo is based on the [planning with diffusion](https://diffusion-planning.github.io/) repository.
The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
