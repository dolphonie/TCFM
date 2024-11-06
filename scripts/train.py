import wandb
import diffuser.utils as utils
import pdb
import torch
from datetime import datetime
import os

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    folder_path=args.datapath,
    horizon=args.horizon,
    history_length=args.history_length,
    include_current=args.include_current,
    predict_features=args.predict_features,
    packed_features=args.packed_features,
    normalization=args.normalization
    )

test_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'test_dataset_config.pkl'),
    folder_path=args.datapath,
    horizon=args.horizon,
    history_length=args.history_length,
    include_current=args.include_current,
    predict_features=args.predict_features,
    packed_features=args.packed_features,
    normalization=args.normalization
    )

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    # env=args.dataset,
)

dataset = dataset_config()
test_dataset = test_dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = 0


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

print(args.global_cond_dim)
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    lstm_in_dim=dataset.packed_dim,
    global_cond_dim = args.global_cond_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
    use_wavelet=args.use_wavelet
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()


diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, test_dataset, renderer)
if args.cont is not None:
    trainer.load_model(args.cont)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

# print('Testing forward...', end=' ', flush=True)
# data, global_cond, conditions = dataset[0]
# data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
# global_cond = torch.tensor(global_cond, dtype=torch.float32).unsqueeze(0)
# loss, _ = diffusion.loss(data, global_cond, [conditions])
# loss.backward()
# print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

