import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer

from functools import partial
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class SidotiAircraftSeparateWavelet(torch.utils.data.Dataset):
    def __init__(self, 
                 folder_path, 
                 horizon,
                 use_wavelet=True
                 ):
        """ Just load the aircraft dataset and predict the future longitude, latitude, and altitude"""
        print("Loading dataset from: ", folder_path)

        # self.observation_dim = 3
        self.horizon = horizon

        self.dones = []
        self.red_locs = []
        self.process_first_time = True
        
        self.max_path_length = 1000
        self.use_padding = True
        self.always_include_one_detection = True

        self._load_data(folder_path)
        self.indices = self.make_indices(self.path_lengths, horizon)
        # print("Shape:", len(self.red_locs))

        self.use_wavelet = use_wavelet
        if self.use_wavelet:
            wavelet = 'haar'
            maxlevel = 1
            wt_mode = 'symmetric'
            device = 'cpu'

            self.dwt = DWT1DForward(wave=wavelet, J=maxlevel, mode=wt_mode).to(device)
            self.idwt = DWT1DInverse(wave=wavelet, mode=wt_mode).to(device)
            self.observation_dim = 4
        else:
            self.dwt = None
            self.idwt = None
            self.observation_dim = 2

    def _load_data(self, file_path):
        self.set_normalization_factors()
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                self._load_file(os.path.join(file_path, file))

    def _load_file(self, file_path):
        df = pd.read_csv(file_path)

        # print("Number of timesteps: ", len(df))
        # red_locs = df[["longitude", "latitude", "altitude"]].values
        red_locs = df[["longitude", "latitude"]].values # just get the longitude and latitude for wavelet
        red_locs = self.normalize(red_locs)
        path_length = len(red_locs)
        # if path_length > self.max_trajectory_length:
        #     raise ValueError("Path length is greater than max trajectory length")

        if self.use_padding:
            red_locs = np.pad(red_locs, ((0, self.horizon), (0, 0)), 'edge')

        if self.process_first_time:
            self.process_first_time = False
            self.red_locs = [red_locs]
            self.path_lengths = [path_length]
        else:
            self.red_locs.append(red_locs)
            self.path_lengths.append(path_length)

        
    def set_normalization_factors(self):
        self.lon_min = -73.91
        self.lon_max = -76.07
        self.lat_min = 40.28
        self.lat_max = 41.55

        self.alt_min = 0
        self.alt_max = 20000


    def normalize(self, arr):
        x = arr[..., 0]
        arr[..., 0] = ((x - self.lon_min) / (self.lon_max - self.lon_min)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.lat_min) / (self.lat_max - self.lat_min)) * 2 - 1

        # z = arr[..., 2]
        # arr[..., 2] = ((z - self.alt_min) / (self.alt_max - self.alt_min)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        if self.use_wavelet and obs.shape[-1] == 4:
            # assert obs.shape[-1] == 4

            if type(obs) == np.ndarray:
                yl = torch.tensor(obs[:, :, :2].transpose(0, 2, 1))
                yh = torch.tensor(obs[:, :, 2:].transpose(0, 2, 1))
            else:
                yl = obs[..., :2].transpose(1, 2)
                yh = obs[..., 2:].transpose(1, 2)
            obs = self.idwt((yl, [yh]))
            obs = obs.transpose(1, 2)

        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.lon_max - self.lon_min) + self.lon_min

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.lat_max - self.lat_min) + self.lat_min

        # z = obs[..., 2]
        # obs[..., 2] = ((z + 1) / 2) * (self.alt_max - self.alt_min) + self.alt_min
        return obs


    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''

        if self.always_include_one_detection:
            s = 1
        else:
            s = 0

        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(s, max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)
        # return self.red_locs.shape[0] - self.num_past_obs*self.num_skip_obs - self.horizon*self.num_skip_obs

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]        
        trajectory = self.red_locs[path_ind][start:end]
        past_obs = torch.tensor(self.red_locs[path_ind][0:start]).float() # use all past observations
        
        cond = (np.array([]), np.array([]))
        batch = (trajectory, past_obs, cond)
        return batch

    def collate_fn_repeat(self):
        return partial(pad_collate_detections_repeat, wavelet_function=self.dwt)

    def collate_fn(self):
        return partial(pad_collate_detections, wavelet_function=self.dwt)

def pad_collate_detections(batch, wavelet_function):
    (data, all_detections, conditions) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0)).float()
    # global_cond = torch.tensor(np.stack(global_cond, axis=0))

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"detections": detections}

    if wavelet_function is not None:
        data = data.transpose(1, 2)
        yl, yh = wavelet_function(data)  # tuple(lo, hi), shape: batch * n_attr * n_sequence
        data = torch.cat([yl, yh[0]], dim=1)
        data = data.transpose(1, 2)


    return data, global_dict, conditions

def pad_collate_detections_repeat(batch, num_samples, wavelet_function):
    (data, all_detections, conditions) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0)).float()
    # global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    # global_cond = global_cond.repeat((num_samples, 1))
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"detections": detections}

    if wavelet_function is not None:
        data = data.transpose(1, 2)
        yl, yh = wavelet_function(data)  # tuple(lo, hi), shape: batch * n_attr * n_sequence
        data = torch.cat([yl, yh[0]], dim=1)
        data = data.transpose(1, 2)

    return data, global_dict, conditions

if __name__ == "__main__":

    # data_path = "/data/prisoner_datasets/sponsor_datasets/processed_sponsor/train"
    # data_path = "/data/prisoner_datasets/october_datasets/4_detect/train"
    # data_path = "/home/sean/october_datasets/4_detect/train"

    # data_path = "/home/sean/october_datasets/3_detect/test"
    # data_path = "/home/sean/october_datasets/7_detect/train"
    # data_path = '/coc/data/prisoner_datasets/Flight Data/flight_track_data_N172CK_2018_subset.csv'
    # data_path = '/home/sean/Flight Data/flight_track_data_N172CK_2018_subset.csv'
    
    data_path = '/coc/data/prisoner_datasets/flight_data/N172CK/2018'

    dataset =  SidotiAircraftSeparate(
                 folder_path = data_path, 
                 horizon = 60)

    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    train_batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn()
    )
