# Evaluate all files in a folder and save them into a text file

## cythonize mujoco-py at first import
import wandb # Just import to relieve the libstdc++ error
import os
import numpy as np
import torch

import tqdm
import diffuser.utils as utils
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
from diffuser.utils.serialization import load_config
from diffuser.datasets.prisoner import pad_collate_detections_repeat
from evaluate import get_stats, render_dist_t, get_stats_batch

def run(loadpath, datapath, n_samples, batch_size, n_iter, condition_start_detect, sample_type):
    # set the seed
    seed = 3
    torch.manual_seed(seed)
    np.random.seed(seed)

    class Args:
        loadpath = None
        diffusion_epoch = "latest"
        n_samples = 4
        device = 'cuda:0'
        
    args = Args()
    args.loadpath = loadpath

    # datapath = '/home/sean/PrisonerEscape/datasets/october_datasets/4_detect_october_50/gnn_map_0_run_50_AStar'
    trainer, dataset = utils.load_model(
        args.loadpath, epoch=args.diffusion_epoch, dataset_path = datapath, condition_path = condition_start_detect)

    model = trainer.ema_model
    renderer = PrisonerRenderer(background=True)

    # start_pad = trainer.start_pad
    start_pad = 0
    print(start_pad)

    def cycle(dl):
        while True:
            for data in dl:
                yield data

    # pad_collate_detections_default
    batch_collate_fn = lambda batch: pad_collate_detections_repeat(batch, n_samples)
    # batch_collate_fn = lambda batch: pad_collate_detections_default(batch, n_samples)

    dataloader_vis = cycle(torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True, collate_fn=batch_collate_fn,
            ))

    render = False

    gt_list = []
    sample_list = []

    for i in tqdm.tqdm(range(n_iter)):
        ## get a single datapoint
        batch = dataloader_vis.__next__()
        # conditions = to_device(batch.conditions, 'cuda:0')

        data, global_cond, conditions = batch
        # for sample_type in ['repaint', 'constrained', 'original']:
        # for sample_type in ['original', 'constrained']:
        # for sample_type in ['original']:

        samples = model.conditional_sample(global_cond, conditions, sample_type=sample_type)
        samples = to_np(samples)

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = samples

        ## [ n_samples x (horizon + 1) x observation_dim ]
        observations = dataset.unnormalize(normed_observations)


        # logdir = '/home/sean/Diffuser/figures/4_detect'
        logdir = './figures/4_detect'
        logdir = os.path.join(logdir, sample_type)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        gt_path = dataset.unnormalize(batch[0])

        if render:
            ground_truth_savepath = os.path.join(logdir, f'{i}_ground_truth.png')
            renderer.composite(ground_truth_savepath, gt_path, ncol=1)

            savepath = os.path.join(logdir, f'{i}_composite_detects.png')
            # renderer.composite(savepath, observations)
            renderer.composite(savepath, observations, conditions=conditions)

            savepath = os.path.join(logdir, f'{i}_composite.png')
            renderer.composite(savepath, observations, conditions=[None] * len(conditions))

            np.save(os.path.join(logdir, f'{i}.npy'), observations)
            np.save(os.path.join(logdir, f'{i}_gt.npy'), gt_path)

        gt_list.append(gt_path)
        sample_list.append(observations)

    total_area = 2428*2428
    dist_min, dist_averages = get_stats_batch(gt_list, sample_list, total_area, batch_size, n_samples)

    print("0: ", np.mean(dist_min[:, 0 + start_pad])/2428, np.std(dist_min[:, 0 + start_pad]/2428))
    print("30: ", np.mean(dist_min[:, 29 + start_pad])/2428, np.std(dist_min[:, 29 + start_pad]/2428))
    print("60: ", np.mean(dist_min[:, 59 + start_pad])/2428, np.std(dist_min[:, 59 + start_pad]/2428))

    print("0: ", np.mean(dist_averages[:, 0 + start_pad])/2428, np.std(dist_averages[:, 0 + start_pad]/2428))
    print("30: ", np.mean(dist_averages[:, 29 + start_pad])/2428, np.std(dist_averages[:, 29 + start_pad]/2428))
    print("60: ", np.mean(dist_averages[:, 59 + start_pad])/2428, np.std(dist_averages[:, 59 + start_pad]/2428))

    return dist_min, dist_averages

if __name__ == "__main__":

    dataset = '3_detect'

    # get all folders in this base folder
    base_folder = f'/data/sye40/prisoner_logs/MRS/base_branch/{dataset}s/best'
    folders = os.listdir(base_folder)
    folders = [os.path.join(base_folder, folder) for folder in folders]

    # datapath = '/data/prisoner_datasets/october_datasets/3_detect_october_50/gnn_map_0_run_50_AStar'
    datapath = f'/data/prisoner_datasets/october_datasets/{dataset}_october_50/gnn_map_0_run_50_AStar'

    n_samples = 30
    batch_size = 20
    start_pad = 0
    n_iter = 100
    condition_start_detect = True
    sample_type = 'original'

    if condition_start_detect:
        cond_str = 'cond'
    else:
        cond_str = 'no_cond'


    for folder in folders:
        # try:
        dist_min, dist_averages = run(folder, datapath, n_samples, batch_size, n_iter, condition_start_detect, sample_type)

        with open(f'save_evals_{dataset}.txt', 'a') as file:
            file.write(f'{folder}\n')
            file.write(f'{np.mean(dist_min[:, 0 + start_pad])/2428}, {np.std(dist_min[:, 0 + start_pad]/2428)}\n')
            file.write(f'{np.mean(dist_min[:, 29 + start_pad])/2428}, {np.std(dist_min[:, 29 + start_pad]/2428)}\n')
            file.write(f'{np.mean(dist_min[:, 59 + start_pad])/2428}, {np.std(dist_min[:, 59 + start_pad]/2428)}\n')
            file.write('\n')

            file.write(f'{np.mean(dist_averages[:, 0 + start_pad])/2428}, {np.std(dist_averages[:, 0 + start_pad]/2428)}\n')
            file.write(f'{np.mean(dist_averages[:, 29 + start_pad])/2428}, {np.std(dist_averages[:, 29 + start_pad]/2428)}\n')
            file.write(f'{np.mean(dist_averages[:, 59 + start_pad])/2428}, {np.std(dist_averages[:, 59 + start_pad]/2428)}\n')
            file.write('\n')

        np.savez(os.path.join(folder, f'distances_{cond_str}_{sample_type}.npz'), dist_min=dist_min, dist_averages=dist_averages)
        # except: 
        #     print(folder)
        #     continue
            
