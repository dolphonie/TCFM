## cythonize mujoco-py at first import
import os
import numpy as np
import torch

import diffuser.utils as utils

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.datasets.prisoner import PrisonerDataset
from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
from diffuser.utils.serialization import load_config
from diffuser.datasets.prisoner import pad_collate_detections
from evaluate import get_stats, render_dist_t

# set the seed
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)


class Args:
#   loadpath = '/data/sye40/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects/prisoner/diffusion/H120_T256'
#   loadpath = '/data/sye40/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects_n_512/prisoner/diffusion/H120_T512'
#   loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects_n_512/prisoner/diffusion/H120_T512'
#   loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects/prisoner/diffusion/H120_T100'
#   loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects/prisoner/diffusion/H120_T100_hideout'
#   loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/prisoner_film_condition_detects/prisoner/diffusion/H120_T100_negative'
  loadpath = '/home/sean/prisoner_logs/diffuser/prisoner/3_detects/prisoner/diffusion/H120_T100'
  diffusion_epoch = "latest"
  n_samples = 4
  device = 'cuda:0'
    
args = Args()

diffusion_experiment = utils.load_diffusion(
    args.loadpath, epoch=args.diffusion_epoch)

dataset_config = load_config(args.loadpath, 'dataset_config.pkl')
# dataset_config._dict['folder_path'] = '/data/prisoner_datasets/october_datasets/4_detect/test'
dataset_config._dict['folder_path'] = '/home/sean/october_datasets/3_detect/train'

dataset = dataset_config()

# renderer_globe = PrisonerRendererGlobe()
# renderer = diffusion_experiment.renderer
renderer = PrisonerRenderer(background=True)
model = diffusion_experiment.trainer.ema_model

def cycle(dl):
    while True:
        for data in dl:
            yield data

# data_path = "/data/prisoner_datasets/october_datasets/4_detect/test"

# dataset = PrisonerDataset(data_path,                  
#             horizon = 256,
#             normalizer = None,
#             preprocess_fns = None,
#             use_padding = False,
#             max_path_length = 40000,
#             dataset_type = "prisoner")

n_samples = 10

dataloader_vis = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True, collate_fn=pad_collate_detections,
        ))

render = True

gt_list = []
sample_list = []

for i in range(5):
    ## get a single datapoint
    batch = dataloader_vis.__next__()
    # conditions = to_device(batch.conditions, 'cuda:0')

    repeated_detects = batch[1]['detections'].data.unsqueeze(0).repeat(n_samples, 1, 1)
    global_cond = {'detections': repeated_detects, 'hideouts': batch[1]['hideouts'].repeat(n_samples, 1)}

    conditions = batch[2] * n_samples
    detected_locs = dataset.detected_locations[1]

    logdir = '/home/sean/Diffuser/figures/gt'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    gt_path = dataset.unnormalize(batch[0])

    print(gt_path.shape)

    if render:
        ground_truth_savepath = os.path.join(logdir, f'{i}_ground_truth.png')
        renderer.composite(ground_truth_savepath, gt_path, ncol=1)

        np.save(os.path.join(logdir, f'{i}_gt.npy'), gt_path)

    gt_list.append(gt_path)

# total_area = 2428*2428
# dist_t = get_stats(gt_list, sample_list, total_area)
# render_dist_t(dist_t)