## cythonize mujoco-py at first import
import os
import copy
import numpy as np
import torch
import einops
import pdb

import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
import pyproj

import math

# set the seed
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

TOTAL_AREA = 0

def evaluate(Args, dataset, renderer, logdir, render=True, n_samples=20, n_batches=10):
        
    args = Args()

    diffusion_experiment = utils.load_diffusion(
        args.loadpath, epoch=args.diffusion_epoch)

    model = diffusion_experiment.trainer.ema_model

    def cycle(dl):
        while True:
            for data in dl:
                yield data

    dataloader_vis = cycle(torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            ))

    # store everything 
    ground_truth_list = []
    sample_list = []

    for i in range(n_batches):
        ## get a single datapoint
        batch = dataloader_vis.__next__()
        # conditions = to_device(batch.conditions, 'cuda:0')

        # print(batch[1])
        global_cond = batch[1].repeat(n_samples, 1)
        conditions = [batch[2]] * n_samples

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = model.conditional_sample(global_cond, conditions)
        samples = to_np(samples)

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = samples

        ## [ n_samples x (horizon + 1) x observation_dim ]
        observations = dataset.unnormalize(normed_observations)

        gt_path = dataset.unnormalize(batch[0])
        ground_truth_list.append(gt_path)
        sample_list.append(observations)

        if render:
            ground_truth_savepath = os.path.join(logdir, f'{i}_ground_truth.png')
            renderer.composite(ground_truth_savepath, gt_path, ncol=1)

            savepath = os.path.join(logdir, f'{i}_composite.png')
            renderer.composite(savepath, observations)

            savepath = os.path.join(logdir, f'{i}_compare.png')
            renderer.composite_samples_gt(savepath, observations, gt_path)

    return ground_truth_list, sample_list

def evaluate_sponsor(render):
    # Evaluate the sponsor dataset
    horizon = 512
    class Args:
        loadpath = '/data/sye40/prisoner_logs/diffuser/store/sponsor_conditioned_H512_T256'
        diffusion_epoch = '160000'
        n_samples = 4
        device = 'cuda:0'

    data_path = "/data/prisoner_datasets/sponsor_datasets/processed_sponsor/test"

    dataset = PrisonerDataset(data_path,                  
                horizon = horizon,
                normalizer = None,
                preprocess_fns = None,
                use_padding = False,
                max_path_length = 40000,
                dataset_type = "sponsor")

    logdir = '/nethome/sye40/diffuser/figures/sponsor/conditional'

    renderer = PrisonerRendererGlobe()

    return evaluate(Args, dataset, renderer, logdir, render)

def evaluate_prisoner(render):
    # Evaluate the lost in the woods
    horizon = 256
    class Args:
        loadpath = '/data/sye40/prisoner_logs/diffuser/store/prisoner_conditioned_H256_T256'
        # loadpath = '/nethome/sye40/diffuser/logs/prisoner/diffusion/H60_T256'
        diffusion_epoch = 'latest'
        n_samples = 4
        device = 'cuda:0'

    data_path = "/data/prisoner_datasets/october_datasets/4_detect/test"

    dataset = PrisonerDataset(data_path,                  
                horizon = horizon,
                normalizer = None,
                preprocess_fns = None,
                use_padding = False,
                max_path_length = 40000,
                dataset_type = "prisoner")

    logdir = '/nethome/sye40/diffuser/figures/conditional_256'

    renderer = PrisonerRenderer(background=True)

    return evaluate(Args, dataset, renderer, logdir, render, n_samples=20)

def evaluate_prisoner_globe(render, n_samples, n_batches):
    horizon = 128
    class Args:
        # loadpath = '/nethome/sye40/diffuser/logs/prisoner/diffusion/H128_T256'
        loadpath = '/data/sye40/diffuser/logs/logs/prisoner/diffusion/H128_T256_conditional'
        diffusion_epoch = 'latest'
        n_samples = 4
        device = 'cuda:0'

    data_path = "/data/prisoner_datasets/sponsor_3_18/test"

    dataset = PrisonerDataset(data_path,                  
                horizon = horizon,
                normalizer = None,
                preprocess_fns = None,
                use_padding = False,
                max_path_length = 40000,
                dataset_type = "prisoner_globe")

    logdir = '/nethome/sye40/diffuser/figures/conditional_globe'

    renderer = PrisonerRendererGlobe()

    return evaluate(Args, dataset, renderer, logdir, render, n_samples=n_samples, n_batches=n_batches)

def get_stats(ground_truth_list, sample_list, total_area):
    total_dists = []
    containments = []
    dist_t = []
    for gt, samples in zip(ground_truth_list, sample_list):
        dists, containment_percentage, dists_timestep= get_ade_confidence_euclidean(gt[0], samples, total_area)
        total_dists.append(dists)
        containments.append(containment_percentage)
        dist_t.append(dists_timestep)
    total_dists = np.concatenate(total_dists, axis=0)
    dist_t = np.concatenate(dist_t, axis=0)

    print("Total distance mean: ", total_dists.mean())
    print("Containment mean: ", np.array(containments).mean())

    print("0: ", np.mean(dist_t[:, 0])/2428, np.std(dist_t[:, 0]/2428))
    print("30: ", np.mean(dist_t[:, 29])/2428, np.std(dist_t[:, 29]/2428))
    print("60: ", np.mean(dist_t[:, 59])/2428, np.std(dist_t[:, 59]/2428))

    return dist_t

def get_stats_batch(ground_truth_list, sample_list, total_area, batch_size, n_samples, mountain_locs=None):
    min_dists = []
    dist_averages = []
    count = 0
    denom = 0
    for gts, samples in zip(ground_truth_list, sample_list):
        if len(gts.flatten()) != n_samples * batch_size * gts.shape[-1] * gts.shape[-2]:
            # Skip due to batch size
            continue

        g = np.reshape(gts, (n_samples, batch_size, gts.shape[-2], gts.shape[-1]))
        s = np.reshape(samples, (n_samples, batch_size, samples.shape[-2], samples.shape[-1]))

        g = np.transpose(g, (1, 0, 2, 3))
        s = np.transpose(s, (1, 0, 2, 3))

        dist_averages.append(np.linalg.norm(gts - samples, axis=-1))

        dist = np.linalg.norm(g - s, axis=-1)

        indices = np.argmin(dist.sum(axis=-1), axis=-1)
        best_dist = dist[np.arange(batch_size), indices]

        # best_dist = np.amin(dist, axis=1)
        # dist = np.linalg.norm(gts - samples, axis=-1)
        min_dists.append(best_dist)

        if mountain_locs is not None:
            for mountain_loc in mountain_locs:
                flattened_arr = samples.reshape(-1, 2)
                mount_dists = np.linalg.norm(flattened_arr - mountain_loc, axis=-1)
                count += np.sum(mount_dists < 140) 
                denom += len(mount_dists)

            print("Percentage of samples in mountain: ", (count/denom) * 100)

    dist_t = np.concatenate(min_dists, axis=0)

    dist_averages = np.concatenate(dist_averages, axis=0)
    return dist_t, dist_averages

def get_stats_batch_geodesic(ground_truth_list, sample_list, total_area, batch_size, n_samples):
    """ Use geodesic distances between trajectories """
    geoconvert = pyproj.Geod(ellps='WGS84')
    min_dists = []
    dist_averages = []
    for gts, samples in zip(ground_truth_list, sample_list):
        g = np.reshape(gts, (n_samples, batch_size, gts.shape[-2], gts.shape[-1]))
        s = np.reshape(samples, (n_samples, batch_size, samples.shape[-2], samples.shape[-1]))

        g = np.transpose(g, (1, 0, 2, 3))
        s = np.transpose(s, (1, 0, 2, 3))

        dist_averages.append(distance(gts, samples, geoconvert))

        # dist_averages.append(np.linalg.norm(gts - samples, axis=-1))

        # dist = np.linalg.norm(g - s, axis=-1)
        dist = distance(g, s, geoconvert)

        indices = np.argmin(dist.sum(axis=-1), axis=-1)
        best_dist = dist[np.arange(batch_size), indices]

        min_dists.append(best_dist)
    dist_t = np.concatenate(min_dists, axis=0)

    dist_averages = np.concatenate(dist_averages, axis=0)
    return dist_t, dist_averages

def distance(p1, p2, geoconvert):
    """ Get distance between two lat/long points"""
    # longitude/latitude

    # longs 1, lats 1, longs 2, lats 2
    fwd_azimuth,back_azimuth,dist = geoconvert.inv(p1[..., 0], p1[..., 1], p2[..., 0], p2[..., 1]) # in meters
    return dist / 1000 # in km

def get_ade_confidence_euclidean(gt, samples, total_area):
    ade = []
    dist_individual = []
    for sample in samples:
        dist = np.linalg.norm(gt - sample, axis=-1)
        dist_individual.append(dist)
    dists = np.concatenate(dist_individual, axis=0)
    dists_timestep = np.stack(dist_individual, axis=0)
    dists = dists[~np.isnan(dists)]

    percentage_area = 0.05 * total_area
    radius = np.sqrt(percentage_area / np.pi)

    containment_percentage = np.sum(dists < radius) / len(dists)
    return dists, containment_percentage, dists_timestep

def get_ade_confidence_geodesic(gt, samples, total_area):
    geoconvert = pyproj.Geod(ellps='WGS84')
    ade = []
    dist_individual = []
    for sample in samples:
        dist = distance(gt, sample, geoconvert)
        dist_individual.append(dist)
    dists = np.concatenate(dist_individual, axis=0)
    dists_timestep = np.stack(dist_individual, axis=0)
    dists = dists[~np.isnan(dists)]

    percentage_area = 0.05 * total_area
    radius = np.sqrt(percentage_area / np.pi)

    containment_percentage = np.sum(dists < radius) / len(dists)
    return dists, containment_percentage, dists_timestep


def compute_area(lon1, lon2, lat1, lat2):
    percentage =  (math.sin(math.radians(lat2))-math.sin(math.radians(lat1))) * math.radians(lon2-lon1) / (4*np.pi)

    earth_radius = 6378 # km
    total_area = 4*np.pi*earth_radius**2
    return percentage * total_area

def get_area(type):
    if type == 'sponsor':
        lon_min = -88.452041 
        lon_max = -40.961581
        lat_min = 10.77808 
        lat_max = 52.470775
    elif type == "prisoner_globe":
        lon_min = -70
        lon_max = 15
        lat_min =  -20
        lat_max = 65
    return compute_area(lon_min, lon_max, lat_min, lat_max)

def render_dist_t(dist_t):
    dist_t = np.mean(dist_t, axis=0)
    x_axis = np.arange(len(dist_t))

    plt.plot(x_axis, dist_t)
    plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
    plt.ylabel("Average Distance Error (ADE)", fontsize=16, fontweight='bold')

    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    plt.savefig("dist_t.png")

if __name__ == "__main__":

    render = False
    total_area = get_area('prisoner_globe')
    print("Total area is: ", total_area)
    ground_truth_list, sample_list = evaluate_prisoner_globe(render, 32, 20)
    # ground_truth_list, sample_list = evaluate_prisoner(render)
    # ground_truth_list, sample_list = evaluate_sponsor(render)
    dist_t = get_stats(ground_truth_list, sample_list, total_area)
    np.save("dist_t_prisoner.npy", dist_t)
    render_dist_t(dist_t)
