import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer

class SidotiAircraft(torch.utils.data.Dataset):
    def __init__(self, 
                 folder_path, 
                 horizon,
                 num_past_obs = 30, # minutes
                 num_skip_obs = 60 # seconds
                 ):
        """ Just load the aircraft dataset and predict the future longitude, latitude, and altitude"""
        print("Loading dataset from: ", folder_path)

        self.observation_dim = 3
        self.horizon = horizon

        self.dones = []
        self.red_locs = []
        self.process_first_graph = True

        self.num_past_obs = num_past_obs
        self.num_skip_obs = num_skip_obs

        self._load_data(folder_path)
        print("Shape:", len(self.red_locs))

    # def make_indices(self, path_lengths, horizon):
    #     '''
    #         makes indices for sampling from dataset;
    #         each index maps to a datapoint
    #     '''
        # indices = np.array([])

        # indices = []
        # for i, path_length in enumerate(path_lengths):
        #     max_start = min(path_length - 1, self.max_path_length - horizon)
        #     if not self.use_padding:
        #         max_start = min(max_start, path_length - horizon)
        #     for start in range(max_start):
        #         end = start + horizon
        #         indices.append((i, start, end))
        # indices = np.array(indices)
        # return indices


    def _load_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(subset = ['longitude', 'latitude', 'altitude'])
        df.loc[df['altitude'] < 0, 'altitude'] = 0

        # Selecting columns 'A', 'B', and 'C'
        selected_columns = df[['longitude', 'latitude', 'altitude']]

        # Converting selected columns to numpy array
        self.red_locs = selected_columns.values

        print("Number of timesteps: ")
        self.set_normalization_factors(df)

        # normalize the data
        self.red_locs = self.normalize(self.red_locs)

    def set_normalization_factors(self, df):
        # self.lon_min = df['longitude'].min()
        # self.lon_max = df['longitude'].max()
        # self.alt_min = df['altitude'].min()

        # self.lat_min = df['latitude'].min()
        # self.lat_max = df['latitude'].max()
        # self.alt_max = df['altitude'].max()
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

        z = arr[..., 2]
        arr[..., 2] = ((z - self.alt_min) / (self.alt_max - self.alt_min)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.lon_max - self.lon_min) + self.lon_min

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.lat_max - self.lat_min) + self.lat_min

        z = obs[..., 2]
        obs[..., 2] = ((z + 1) / 2) * (self.alt_max - self.alt_min) + self.alt_min
        return obs

    # def _load_file(self, file):

    #     timesteps = file["timestep_observations"]
    #     if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
    #         # prisoner dataset
    #         detected_locations = file["detected_locations"]
    #         red_locs = np.float32(file["red_locations"])
    #         hideout_locs = np.float32(file["hideout_observations"])[0]
    #     else:
    #         # sponsor dataset
    #         detected_locations = file["all_detections_of_fugitive"]
    #         red_locs = np.float32(file["red_locations"])
        
    #     path_length = len(red_locs)
    #     if path_length > self.max_trajectory_length:
    #         raise ValueError("Path length is greater than max trajectory length")

    #     if self.global_lstm_include_start:
    #         detected_locations[0] = copy.deepcopy(file["red_locations"][0]) / 2428

    #     if self.process_first_graph:
    #         self.process_first_graph = False
    #         self.timesteps = timesteps
    #         self.dones = file["dones"]
    #         self.red_locs = [red_locs]
    #         self.detected_locations = [detected_locations]
    #         self.path_lengths = [path_length]
    #         if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
    #             self.hideout_locs = [hideout_locs]
    #     else:
    #         self.red_locs.append(red_locs)
    #         self.timesteps = np.append(self.timesteps, timesteps)
    #         self.dones = np.append(self.dones, file["dones"])
    #         self.detected_locations.append(detected_locations)
    #         self.path_lengths.append(path_length)
    #         if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
    #             self.hideout_locs.append(hideout_locs)

    # def process_detections(self):
    #     self.detected_dics = []
    #     for detected_locs in self.detected_locations:
    #         indices = []
    #         detects = []
    #         for i in range(len(detected_locs)):
    #             loc = detected_locs[i]
    #             if self.dataset_type == 'sidoti':
    #                 if loc[0] != -np.inf:
    #                     loc[0] = (loc[0] - self.min_x) / (self.max_x - self.min_x)
    #                     loc[1] = (loc[1] - self.min_y) / (self.max_y - self.min_y)
    #                     indices.append(i)
    #                     detects.append(loc)
    #             else:
    #                 if loc[0] != -1:
    #                     if self.dataset_type == "sponsor" or self.dataset_type == "prisoner_globe":
    #                         # sponsor dataset needs normalization
    #                         loc[0] = (loc[0] - self.min_x) / (self.max_x - self.min_x)
    #                         loc[1] = (loc[1] - self.min_y) / (self.max_y - self.min_y)
    #                     elif self.dataset_type == "prisoner":
    #                         # need to convert from 0-1 to -1 to 1
    #                         loc[0] = loc[0] * 2 - 1
    #                         loc[1] = loc[1] * 2 - 1
    #                     indices.append(i)
    #                     detects.append(loc)
    #         detects = np.stack(detects, axis=0)
    #         indices = np.stack(indices, axis=0)
    #         self.detected_dics.append((indices, detects))

    # def _preprocess_detections(self, detected_locs, timestamps):
    #     """ Given a numpy array of [T x 2] where if there is a detection, the value is (x, y) and if there is not, the value is (-1, -1)
        
    #     For each row in the array, return all previous detections before that row
    #     Also need to add the time difference between each step so we return a [dt, x, y] for each detection
    #     """
    #     processed_detections = []
    #     detected_locs = self.coordinate_transform(detected_locs)
    #     detected_locs = np.concatenate((detected_locs, timestamps), axis=1)
    #     for i in range(detected_locs.shape[0]):
    #         curr_detections = copy.deepcopy(detected_locs[:i+1])
    #         curr_detections = curr_detections[curr_detections[:, 0] != -1]
    #         curr_detections[:, 2] = detected_locs[i, 2] - curr_detections[:, 2]
    #         processed_detections.append(curr_detections)
    #     return processed_detections

    # def get_conditions(self, idx, start, end, trajectories):
    #     '''
    #         condition on current observation for planning
    #     '''
    #     detected_dic = self.detected_dics[idx]
    #     # subtract off the start and don't take anything past the end

    #     # self.end_pad is used to ensure that we have no detections in this region.
    #     # This is so we can call this part the prediction region
    #     start_idx_find = np.where(detected_dic[0] >= start)[0]
    #     end_idx_find = np.where(detected_dic[0] < end - self.end_pad)[0]

    #     # These are global conditions where the global_cond_idx is the 
    #     # integer index within the trajectory of where the detection occured

    #     # Take the detections before the start of the trajectory
    #     before_start_detects = np.where(detected_dic[0] < end - self.end_pad)[0]
    #     if len(before_start_detects) == 0:
    #         global_cond_idx = np.array([])
    #         global_cond = np.array([])
    #     else:
    #         global_cond_idx = detected_dic[0][:before_start_detects[-1]]
    #         global_cond = detected_dic[1][:before_start_detects[-1]]

    #     detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, end - self.end_pad)

    #     if self.condition_path:
    #         if len(start_idx_find) == 0 or len(end_idx_find) == 0 or start_idx_find[0] > end_idx_find[-1]:
    #             # always include the start of the path
    #             if self.include_start_detection:
    #                 idxs = np.array([0])
    #                 detects = np.array([trajectories[0]])
    #             else:
    #                 idxs = np.array([])
    #                 detects = np.array([])
    #         else:
    #             start_idx = start_idx_find[0]
    #             end_idx = end_idx_find[-1]

    #             idxs = detected_dic[0][start_idx:end_idx+1] - start
    #             detects = detected_dic[1][start_idx:end_idx+1]

    #             if idxs[0] != 0 and self.include_start_detection:
    #                 idxs = np.concatenate((np.array([0]), idxs))
    #                 detects = np.concatenate((np.array([trajectories[0]]), detects))
    #     else:
    #         idxs = np.array([])
    #         detects = np.array([])

    #     return detection_lstm, (idxs, detects)

    # def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
    #     """ Convert the indices back to timesteps and concatenate them together"""
    #     detection_num = min(self.max_detection_num, len(global_cond_idx))
    #     global_cond_idx = global_cond_idx[-detection_num:]
    #     global_cond = global_cond[-detection_num:]

    #     # no detections before start, just pad with -1, -1
    #     if len(global_cond_idx) == 0:
    #         return torch.tensor([[-1, -1, -1]])

    #     # convert the indices back to timesteps
    #     global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
    #     global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
    #     return torch.tensor(global_cond)

    def __len__(self):
        # return len(self.indices)
        return self.red_locs.shape[0] - self.num_past_obs*self.num_skip_obs - self.horizon*self.num_skip_obs

    def __getitem__(self, idx):
        
        end_past_idx = idx + self.num_past_obs*self.num_skip_obs
        end_idx = end_past_idx + self.horizon*self.num_skip_obs

        past_obs = self.red_locs[idx:end_past_idx:self.num_skip_obs]
        trajectory = self.red_locs[end_past_idx:end_idx:self.num_skip_obs]
        cond = (np.array([]), np.array([]))
        batch = (trajectory, past_obs, cond)
        return batch

    def collate_fn_repeat(self):
        return pad_collate_repeat

    def collate_fn(self):
        return pad_collate

def pad_collate_repeat(batch, num_samples):
    (data, global_cond, cond) = zip(*batch)
    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0)).float()
    # cond = torch.tensor(np.stack(cond, axis=0))

    data = data.repeat(num_samples, 1, 1)
    global_cond = global_cond.repeat(num_samples, 1, 1)
    return data.float(), {"detections": global_cond}, cond * num_samples

def pad_collate(batch):
    (data, global_cond, cond) = zip(*batch)
    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0)).float()
    # cond = torch.tensor(np.stack(cond, axis=0))
    return data.float(), {"detections": global_cond}, cond

# def pad_collate_detections(batch):c
#     (data, global_cond, all_detections, conditions) = zip(*batch)

#     data = torch.tensor(np.stack(data, axis=0))
#     global_cond = torch.tensor(np.stack(global_cond, axis=0))

#     x_lens = [len(x) for x in all_detections]
#     xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
#     detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

#     # Pass this to condition our models rather than pass them separately
#     global_dict = {"hideouts": global_cond, "detections": detections}

#     return data, global_dict, conditions

# def pad_collate_global(batch):
#     (data, global_cond, conditions) = zip(*batch)
    
#     data = torch.tensor(np.stack(data, axis=0))
#     global_cond = torch.tensor(np.stack(global_cond, axis=0))

#     return data, global_cond, conditions

if __name__ == "__main__":

    # data_path = "/data/prisoner_datasets/sponsor_datasets/processed_sponsor/train"
    # data_path = "/data/prisoner_datasets/october_datasets/4_detect/train"
    # data_path = "/home/sean/october_datasets/4_detect/train"

    # data_path = "/home/sean/october_datasets/3_detect/test"
    # data_path = "/home/sean/october_datasets/7_detect/train"
    # data_path = '/coc/data/prisoner_datasets/Flight Data/flight_track_data_N172CK_2018_subset.csv'
    data_path = '/home/sean/Flight Data/flight_track_data_N172CK_2018_subset.csv'

    dataset =  SidotiAircraft(
                 folder_path = data_path, 
                 horizon = 60,
                 num_past_obs=10)
    
    print(dataset[0])

    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    train_batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn()
    )

    # for i, data in enumerate(cycle(dataloader)):
    #     # global_dict = data[1]
    #     print(data[0].shape)
    #     print(data[1])
    #     # print(data[2].shape)
    #     break

    # # print(dataset.path_lengths)
    # gt_path = dataset[0][1]
    # print(type(gt_path))
    # print(dataset[0][0].shape)
    # print(gt_path.shape)