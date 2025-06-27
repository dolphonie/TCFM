import os
import numpy as np
import torch
import tqdm
import wandb
import diffuser.utils as utils
from diffuser.utils.arrays import to_np, to_device
from diffuser.utils.rendering import SidotiMPLRenderer, PrisonerRendererGlobe
from evaluate import get_stats_batch_geodesic

# Set the seed for reproducibility
SEED = 5
torch.manual_seed(SEED)
np.random.seed(SEED)

class Args:
    loadpath = '/home/pdkao_google_com/wing-involi-autopilot-data/20250625-2215/'
    # loadpath = '/coc/data/sye40/prisoner_logs/aircraft_sidoti_weather/cfm/H60_T100/20240828-1909'
    # loadpath = '/coc/data/sye40/prisoner_logs/aircraft_sidoti_weather/cfm/H60_T100/20240829-1857/'
    # diffusion_epoch = "latest"
    diffusion_epoch = 14000
    n_samples = 4
    device = 'cuda:0'

def cycle(dl):
    while True:
        for data in dl:
            yield data

def main():
    args = Args()
    datapath = '/home/pdkao_google_com/TCFM/data/val_involi_fixed_subset'
    
    # Load model and dataset
    trainer, dataset = utils.load_model(args.loadpath, epoch=args.diffusion_epoch, dataset_path=datapath)
    model = trainer.ema_model
    # renderer = SidotiMPLRenderer()
    renderer = PrisonerRendererGlobe()

    # Setup data loading
    n_samples = 10  # Reduced for rendering
    batch_size = 1  # Set to 1 for rendering
    render = True  # Flag to enable rendering

    collate_fn = dataset.collate_fn_repeat
    batch_collate_fn = lambda batch: collate_fn(batch, n_samples)
    dataloader_vis = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=True, 
        pin_memory=True, collate_fn=batch_collate_fn,
    ))

    gt_list = []
    sample_list = []

    for i in tqdm.tqdm(range(20)):
        batch = next(dataloader_vis)
        data, global_cond, conditions = batch

        sample_type = 'original'
        samples = model.conditional_sample(global_cond, conditions, sample_type=sample_type)
        samples = to_np(samples)

        observations = dataset.unnormalize(samples)
        gt_path = dataset.unnormalize(batch[0])

        gt_list.append(gt_path)
        sample_list.append(observations)

        if render:
            render_sample(batch, renderer, i, gt_path, observations, conditions)

    # Evaluate results
    total_area = 2428 * 2428
    dist_min, dist_averages = get_stats_batch_geodesic(gt_list, sample_list, total_area, batch_size, n_samples)

    print_results(dist_min, dist_averages)
    save_results(args.loadpath, dist_min, dist_averages)

def render_sample(batch, renderer, index, gt_path, observations, conditions):
    logdir = './figures/involi/'
    os.makedirs(logdir, exist_ok=True)

    # Render ground truth
    gt_render = np.expand_dims(gt_path[0], 0)
    ground_truth_savepath = os.path.join(logdir, f'{index}_ground_truth.png')
    renderer.composite(ground_truth_savepath, gt_render, ncol=1)

    # Render samples
    savepath = os.path.join(logdir, f'{index}_composite.png')
    renderer.composite(savepath, observations, conditions=[None] * len(conditions))

    # Save numpy arrays
    np.save(os.path.join(logdir, f'{index}.npy'), observations)
    np.save(os.path.join(logdir, f'{index}_gt.npy'), gt_path)
    np.save(os.path.join(logdir, f'{index}_batch.npy'), batch)

def print_results(dist_min, dist_averages):
    for t in [0, 29, 59]:
        print(f"{t}: {np.mean(dist_min[:, t]):.2f} ± {np.std(dist_min[:, t]):.2f} (min)")
        print(f"{t}: {np.mean(dist_averages[:, t]):.2f} ± {np.std(dist_averages[:, t]):.2f} (avg)")

def save_results(loadpath, dist_min, dist_averages):
    np.savez(os.path.join(loadpath, 'distances_cond.npz'), dist_min=dist_min, dist_averages=dist_averages)

if __name__ == "__main__":
    main()
