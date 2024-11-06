import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.ConditionalUnet1D',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 120,
        'global_cond_dim': 3,
        'n_diffusion_steps': 100,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.PrisonerRenderer',
        # 'noise_amplitude': 0,

        ## dataset
        'loader': 'datasets.PrisonerDatasetDetections',
        # 'datapath': '/home/sean/PrisonerEscape/datasets/october_datasets/4_detect/train',
        # 'test_datapath': '/home/sean/PrisonerEscape/datasets/october_datasets/4_detect/test',
        'datapath': '/coc/data/prisoner_datasets/october_datasets/4_detect/train',
        'test_datapath': '/coc/data/prisoner_datasets/october_datasets/4_detect/test',
        # 'datapath': '/home/sean/october_datasets/4_detect/train',
        # 'test_datapath': '/home/sean/october_datasets/4_detect/test',
        'dataset_type': 'prisoner',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 40000,
        'global_lstm_include_start': True,
        'condition_path': False,
        'max_trajectory_length': 4320,
        'noise_amplitude': 0.0,
        'cont': None,
        'use_wavelet': False,

        ## serialization
        # 'logbase': '/home/sean/prisoner_logs/diffuser/prisoner/4_detects',
        'logbase': '/coc/data/prisoner_logs/diffuser/prisoner/4_detects',
        # 'logbase': '/home/sean/prisoner_logs/diffuser/prisoner/4_detects',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 2000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 60,
        'n_diffusion_steps': 100,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}