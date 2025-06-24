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
        'diffusion': 'models.CFM',
        'horizon': 60,
        'global_cond_dim': 0,
        'n_diffusion_steps': 100,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.PrisonerRendererGlobe',
        'num_past_obs': 10,
        ## dataset
        'loader': 'datasets.GeneralTrajectoryDataset',
        'use_wavelet': False,
        # 'datapath': '/home/sean/flight_data/N172CK/converted_train',
        # 'test_datapath': '/home/sean/flight_data/N172CK/converted_train',
        'datapath': '/home/pdkao_google_com/TCFM/data/train',
        'test_datapath': '/home/pdkao_google_com/TCFM/data/test',
        'clip_denoised': True,
        'cont': None,
        ## dataset specific configuration
        'history_length': 10,
        'include_current': True,
        'predict_features': ['longitude', 'latitude', 'altitude'],
        # 'packed_features': ['stapac_sfc', 'airtmp_sig', 'lndsea_sfc', 'relhum_sig', 'terrht_sfc', 'trpres_sfc', 'ttlprs_sig', 'uutrue_sig', 'vvtrue_sig', 'turbke_sig', 'longitude', 'latitude', 'altitude'],
        'packed_features': ['longitude', 'latitude', 'altitude'],
        'normalization': {
            'stapac_sfc': {'min': -0.08, 'max': 0.09},
            'airtmp_sig': {'min': 0.84, 'max': 13.89},
            'lndsea_sfc': {'min': 0.05, 'max': 1.06},
            'relhum_sig': {'min': 55.69, 'max': 95.63},
            'terrht_sfc': {'min': -59.95, 'max': 193.30},
            'trpres_sfc': {'min': 993.17, 'max': 1024.91},
            'ttlprs_sig': {'min': 965.97, 'max': 1046.39},
            'uutrue_sig': {'min': -2.59, 'max': 4.75},
            'vvtrue_sig': {'min': -3.19, 'max': 3.76},
            'turbke_sig': {'min': -0.28, 'max': 1.14},
            'longitude': {'min': -88.00, 'max': -69.97},
            'latitude': {'min': 30.74, 'max': 43.83},
            'altitude': {'min': -100.00, 'max': 44000.00},
        },
        ## serialization
        # 'logbase': '/home/sean/prisoner_logs/aircraft_sidoti_last_n_IROS24/',
        'logbase': '/coc/data/sye40/prisoner_logs/aircraft_sidoti_weather/',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),
        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 500000,
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