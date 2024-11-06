import wandb
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer

def get_lowest_root_folders(root_folder):
    lowest_folders = []
    
    # Get all items in the root folder
    items = os.listdir(root_folder)
    
    # Check if each item is a directory
    for item in items:
        item_path = os.path.join(root_folder, item)
        
        if os.path.isdir(item_path):
            # Recursively call the function for subfolders
            subfolders = get_lowest_root_folders(item_path)
            
            if not subfolders:
                # If there are no subfolders, add the current folder to the lowest_folders list
                lowest_folders.append(item_path)         
            lowest_folders.extend(subfolders)
    if len(lowest_folders) == 0:
        return [root_folder]
    return lowest_folders

class PrisonerLanguage(torch.utils.data.Dataset):
    def __init__(self, 
                 folder_path, 
                 horizon,
                 normalizer,
                 preprocess_fns,
                 use_padding,
                 max_path_length,
                 dataset_type = "prisoner_language",
                 include_start_detection = False,
                 global_lstm_include_start = False,
                 condition_path = True,
                 max_detection_num = 32,
                 max_trajectory_length = 4320,
                 end_pad = 60,
                 null_prob = 0.2):
        print("Loading dataset from: ", folder_path)

        self.global_lstm_include_start = global_lstm_include_start
        self.condition_path = condition_path

        self.dataset_type = dataset_type
        self.use_padding = use_padding
        self.observation_dim = 2
        self.horizon = horizon
        self.max_detection_num = max_detection_num
        self.max_trajectory_length = max_trajectory_length
        self.end_pad = end_pad

        self.dones = []
        self.red_locs = []
        self.process_first_graph = True

        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]
        self.max_path_length = max_path_length
        self.include_start_detection = include_start_detection
        self.indices = self.make_indices(self.path_lengths, horizon)

        self.null_prob = null_prob

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def _load_data(self, folder_path):

        np_files = []
        fps = get_lowest_root_folders(folder_path)
        # print(fps)
        for i, fp in enumerate(fps):
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                np_files.append((np_file, i))

        for np_file, i in np_files:
            if np_file["red_locations"].shape[0] != 4320: # skip whatever happened here
                self._load_file(np_file, i)

        print("Path Lengths: ")
        print(max(self.path_lengths), min(self.path_lengths))

        self.set_normalization_factors()
        for i in range(len(self.red_locs)):
            self.red_locs[i] = self.normalize(self.red_locs[i])
        # self.process_detections()

        # after processing detections, we can pad
        if self.use_padding:
            for i in range(len(self.red_locs)):
                # need to add padding to the end of the red_locs
                self.red_locs[i] = np.pad(self.red_locs[i], ((0, self.horizon), (0, 0)), 'edge')
        
        # # normalize hideout locations
        # if self.dataset_type == "prisoner_globe":
        #     for i in range(len(self.hideout_locs)):
        #         self.hideout_locs[i] = self.normalize(self.hideout_locs[i])

    def set_normalization_factors(self):
        # if self.dataset_type == "sponsor" or self.dataset_type == "prisoner_globe" or self.dataset_type == 'sidoti':
        #     print(self.red_locs)
            
        #     all_red_locs = np.concatenate(self.red_locs, axis=0)

        #     self.min_x = min(all_red_locs[:, 0])
        #     self.max_x = max(all_red_locs[:, 0])
        #     self.min_y = min(all_red_locs[:, 1])
        #     self.max_y = max(all_red_locs[:, 1])
        if self.dataset_type == "prisoner_language":
            self.min_x = -90
            self.max_x = -30
            self.min_y = 0
            self.max_y = 60
        else:
            self.min_x = 0
            self.max_x = 2428
            self.min_y = 0
            self.max_y = 2428

    def normalize(self, arr):
        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return obs

    def _load_file(self, file, traj_class):

        # timesteps = file["timestep_observations"]
        red_locs = np.float32(file["red_locations"]).squeeze()
        timesteps = np.arange(red_locs.shape[0]) / self.max_trajectory_length
        
        path_length = len(red_locs)
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        # if self.global_lstm_include_start:
        #     detected_locations[0] = copy.deepcopy(file["red_locations"][0]) / 2428

        if self.process_first_graph:
            self.process_first_graph = False
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = [red_locs]
            self.traj_class = [traj_class]
            self.sentence = [file["sentence"]]
            # self.detected_locations = [detected_locations]
            self.path_lengths = [path_length]
            # if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
            #     self.hideout_locs = [hideout_locs]
        else:
            self.red_locs.append(red_locs)
            self.traj_class.append(traj_class)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.sentence.append(file["sentence"])
            # self.detected_locations.append(detected_locations)
            self.path_lengths.append(path_length)
            # if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
            #     self.hideout_locs.append(hideout_locs)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        trajectories = self.red_locs[path_ind][start:end]
        # conditions = self.get_conditions(trajectories)
        # all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)

        # sample from null probability, if null, randomly set class to -1
        if np.random.rand() < self.null_prob:
            traj_class = -1
        else:
            traj_class = self.traj_class[path_ind]
            timestep = start / self.max_trajectory_length
            

        if np.random.rand() < self.null_prob:
            timestep = -1
        else:
            timestep = start / self.max_trajectory_length
        
        if np.random.rand() < self.null_prob:
            start_location = np.array([-2, -2]) 
        else:
            start_location = self.red_locs[path_ind][start]

        traj_class = np.array([traj_class], dtype=float)
        timestep = np.array([timestep], dtype=float)

        # global_cond = np.concatenate((traj_class, timestep, start_location))
        # hideout_loc = self.hideout_locs[path_ind]
        # global_cond = np.concatenate((hideout_loc, np.array([timestep], dtype=float)))
        batch = trajectories
        return batch, self.sentence[path_ind], timestep
    
    def collate_fn(self):
        return pad_collate
    
    def collate_fn_repeat(self):
        return pad_collate_repeat

def pad_collate(batch):
    (data, sentence, timestep) = zip(*batch)
    data = torch.tensor(np.stack(data, axis=0))

    sentences = [str(s) for s in sentence]
    timesteps = torch.tensor(np.stack(timestep, axis=0)).float()
    # global_cond = np.stack(global_cond)

    cond = [([], [])] * data.shape[0]

    return data, {'sentences': sentences, 'timestep': timesteps}, cond

def pad_collate_repeat(batch, num_samples):
    (data, sentence, timestep) = zip(*batch)
    data = torch.tensor(np.stack(data, axis=0))

    data = data.repeat(num_samples, 1, 1)

    # global_cond = np.stack(global_cond)
    # global_cond = torch.tensor(global_cond).float()
    # global_cond = global_cond.repeat(num_samples, 1)

    sentences = [str(s) for s in sentence] * num_samples
    timesteps = torch.tensor(np.stack(timestep, axis=0)).float()
    timesteps = timesteps.repeat(num_samples, 1)

    cond = [([], [])] * data.shape[0]

    return data, {'sentences': sentences, 'timestep': timesteps}, cond

if __name__ == "__main__":
    data_path = '/data/prisoner_datasets/language_v2/map_0_run_400_RRT'

    dataset = PrisonerLanguage(data_path,                  
                 horizon = 120,
                 normalizer = None,
                 preprocess_fns = None,
                 use_padding = True,
                 max_path_length = 40000,
                 dataset_type = "prisoner_language",
                 include_start_detection=False,
                 condition_path = False,
                 end_pad = 60,
                 max_trajectory_length = 1000)

    # print(len(dataset))
    # print(dataset[0])

    # test pad_collate function
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn())
    for batch in dataloader:
        print(batch)
        break