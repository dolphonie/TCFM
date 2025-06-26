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
        'datapath': '/home/pdkao_google_com/TCFM/data/train_involi_fixed_subset',
        'test_datapath': '/home/pdkao_google_com/TCFM/data/val_involi_fixed',
        'clip_denoised': True,
        'cont': None,
        ## dataset specific configuration
        'history_length': 10,
        'include_current': True,
        'predict_features': ['timestamp', 'long', 'lat', 'alt'],
        # 'packed_features': ['stapac_sfc', 'airtmp_sig', 'lndsea_sfc', 'relhum_sig', 'terrht_sfc', 'trpres_sfc', 'ttlprs_sig', 'uutrue_sig', 'vvtrue_sig', 'turbke_sig', 'long', 'lat', 'alt'],
        'packed_features': ['timestamp', 'long', 'lat', 'alt'],
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
            'long': {'min': -174.51438868550943, 'max': 179.9995279937459},
            'lat': {'min': -86.99611902236938, 'max': 89.509804},
            'alt': {'min': -17598.105439005703, 'max': 206041.90305850637},
            'timestamp': {'min': 1708081917, 'max': 1781959757}
        },
        ## serialization
        # 'logbase': '/home/sean/prisoner_logs/aircraft_sidoti_last_n_IROS24/',
        'logbase': '/home/pdkao_google_com/TCFM/train_logs/',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),
        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 500000,
        'batch_size': 98304,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 2000,
        'n_saves': 500,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },
    'plan': {
        'batch_size': 16,
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
