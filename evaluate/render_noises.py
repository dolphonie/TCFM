## cythonize mujoco-py at first import
import wandb
import os
import numpy as np
import torch

import diffuser.utils as utils

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
from diffuser.utils.serialization import load_config
from diffuser.datasets.prisoner import pad_collate_detections, pad_collate_detections_repeat
from evaluate import get_stats, render_dist_t
import tqdm

# set the seed
seed = 5
torch.manual_seed(seed)
np.random.seed(seed)

class Args:
    # loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/4_detects/conditioning/condition/diffusion/H120_T100/20230612-2306'
    # loadpath = '/data/prisoner_logs/diffuser/prisoner/4_detects/agent_gnn/diffusion/H120_T100/20230614-1427'
    # loadpath = '/data/sye40/prisoner_logs/diffuser/prisoner/4_detects/20230610-1611'
    loadpath = '/data/sye40/prisoner_logs/diffuser/prisoner/7_detects/20230611-1700'
    diffusion_epoch = "latest"
#   loadpath = '/data/sye40/prisoner_logs/diffuser/prisoner/random/prisoner/diffusion/H120_T100'
#   diffusion_epoch = "latest"
    n_samples = 4
    device = 'cuda:0'
    
args = Args()

# diffusion_experiment = utils.load_diffusion(
#     args.loadpath, epoch=args.diffusion_epoch)

# dataset_path = '/home/sean/PrisonerEscape/datasets/october_datasets/4_detect_october_50/gnn_map_0_run_50_AStar'
datapath = '/data/prisoner_datasets/october_datasets/7_detect_october_50/gnn_map_0_run_50_AStar'

# dataset_path = '/data/prisoner_datasets/october_datasets/4_detect_october_50/gnn_map_0_run_50_AStar'
trainer, dataset = utils.load_model(args.loadpath, epoch = args.diffusion_epoch, dataset_path = datapath)

renderer = PrisonerRenderer(background=True)
model = trainer.ema_model

def cycle(dl):
    while True:
        for data in dl:
            yield data

n_samples = 1
batch_size = 1

batch_collate_fn = lambda batch: pad_collate_detections_repeat(batch, n_samples)
dataloader_vis = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True, collate_fn=batch_collate_fn,
        ))
render = True

gt_list = []
sample_list = []

# for i in tqdm.tqdm(range(10)):
    ## get a single datapoint
batch = dataloader_vis.__next__()

data, global_cond, conditions = batch

# for sample_type in ['repaint', 'constrained', 'original']:
# for sample_type in ['original', 'constrained']:
for sample_type in ['constrained']:

    _, all_samples = model.conditional_sample(global_cond, conditions, sample_type=sample_type)
    samples = to_np(all_samples)

    ## [ n_samples x horizon x observation_dim ]
    normed_observations = samples[0]

    ## [ n_samples x (horizon + 1) x observation_dim ]
    observations = dataset.unnormalize(normed_observations)

    # logdir = '/home/sean/Diffuser/figures/4_detect'
    logdir = './figures/7_detect/mountain/render/'
    logdir = os.path.join(logdir, sample_type)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for i in range(observations.shape[0]):
        ground_truth_savepath = os.path.join(logdir, f'{i}.png')
        renderer.composite(ground_truth_savepath, observations[i][np.newaxis, :], ncol=1)

# sample_list.append(observations)

# total_area = 2428*2428
# dist_t = get_stats(gt_list, sample_list, total_area)
# render_dist_t(dist_t)