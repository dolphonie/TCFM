import numpy as np
import torch
from torch import nn
import pdb
import copy
from torchcfm.conditional_flow_matching import *
from torchdyn.core import NeuralODE
import torchdiffeq
# from torchcfm.models.unet import UNetModel

from diffuser.models.helpers import get_schedule_jump

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

def smoothness_loss(traj):
    smoothness_out = (traj[:, :-1, :] - traj[:, 1:, :])**2
    smoothness_out = torch.sqrt(smoothness_out.sum(dim=-1))
    return smoothness_out.mean()

def constraint_loss(traj, conditions):
    c_loss = 0
    for b, conditions in enumerate(conditions):
        if len(conditions[0]) > 0: # if there are conditions
            # traj[b, conditions[0]] = torch.tensor(conditions[1], dtype=x.dtype).to(x.device)
            d = (traj[b, conditions[0]] - torch.tensor(conditions[1], dtype=traj.dtype).to(traj.device)) ** 2
            d = torch.sqrt(d.sum(dim=-1))
            c_loss += d.mean()
    return c_loss

def mountain_loss(traj):
    m_center = np.array([[1800, 1600]])/ 2428. 
    m_center = m_center * 2 - 1
    m_center = torch.tensor(m_center, dtype=traj.dtype).to(traj.device)

    m_center = m_center.unsqueeze(0)
    m_center = m_center.repeat((traj.shape[0], traj.shape[1], 1))
    m_radius = (80 / 2428.) * 2

    d = torch.sqrt(((traj - m_center) ** 2).sum(dim=-1))
    loss = - d / m_radius + 1
    loss = torch.clamp(loss, min=0)

    return loss.mean()

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, use_wavelet=False
    ):
        super().__init__()
        if use_wavelet:
            self.horizon = horizon // 2
        else:
            self.horizon = horizon

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        sigma = 0.0
        # self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # self.register_buffer('posterior_variance', posterior_variance)

        # ## log calculation clipped because the posterior variance
        # ## is 0 at the beginning of the diffusion chain
        # self.register_buffer('posterior_log_variance_clipped',
        #     torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # self.register_buffer('posterior_mean_coef1',
        #     betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # self.register_buffer('posterior_mean_coef2',
        #     (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # ## get loss coefficients and initialize objective
        # loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def set_sampling_timesteps(self, t):
        self.n_timesteps = t

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, global_cond):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, global_cond=global_cond))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_estimate_feature(self, x, cond, t, global_cond):
        log_p_x = self.model(x, t, global_cond=global_cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=log_p_x)
        global_cond_class = torch.clone(global_cond['class'])
        global_cond_class[:, 0] = 0.0 
        log_p_x_given_a = self.model(x, t, global_cond={'class': global_cond_class})
        global_cond_class[:, 0] = 1.0 
        log_p_x_given_b = self.model(x, t, global_cond={'class': global_cond_class})

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        multiplier = -1 / (extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape))
        batch_size = x.shape[0]
        log_p_a_given_x = multiplier * torch.linalg.vector_norm((log_p_x_given_a - log_p_x), dim=(1, 2))
        log_p_b_given_x = multiplier * torch.linalg.vector_norm((log_p_x_given_b - log_p_x), dim=(1, 2))

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, log_p_a_given_x, log_p_b_given_x


    @torch.no_grad()
    def p_sample(self, x, global_cond, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_estimate_feature(self, x, global_cond, cond, t):
        """ Use the repaint sampler to condition on the known timesteps """
        b, *_, device = *x.shape, x.device
        noise= torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
        model_mean, _, model_log_variance, p_a_given_x, p_b_given_x = self.p_mean_variance_estimate_feature(x=x, cond=cond, t=t, global_cond=global_cond)
        model_mean = model_mean.clone().detach()
        model_mean.requires_grad_()
        
        # optimizer = torch.optim.Adam([model_mean], lr=1e-2)
        # losses = []
        # for i in range(8):
        #     optimizer.zero_grad()
        #     l = 7*mountain_loss(model_mean) + 50 * smoothness_loss(model_mean)
        #     losses.append(l.item())
        #     l.backward()
        #     optimizer.step()

        new_model_mean = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise    

        return new_model_mean, p_a_given_x, p_b_given_x


    @torch.no_grad()
    def p_sample_repaint(self, x, global_cond, cond, t, noise=None, device='cuda'):
        """ Use the repaint sampler to condition on the known timesteps """
        # global cond are {detections and hideouts}
        # cond are the detections that are within the current plan
        cond_noised = copy.deepcopy(cond)

        # add noise to the ground truth detections
        for i in range(len(cond_noised)):
            if isinstance(cond_noised[i][1], (np.ndarray, np.generic)):
                data = torch.from_numpy(cond_noised[i][1]).to(device)
            else:
                data = cond_noised[i][1].to(device)
            # t_full = torch.ones(data.size(0)).to() * t[i]
            t_full = torch.full((data.size(0),), t[i], device=device, dtype=torch.long)
            if noise is None:
                noise = torch.randn_like(data).to(device)
            data_noised =  (
                extract(self.sqrt_alphas_cumprod, t_full, data.shape) * data +
                extract(self.sqrt_one_minus_alphas_cumprod, t_full, data.shape) * noise
            )
            cond_noised[i][1] = data_noised

        # apply the new conditioning to the trajectory
        # this essentially masks our x
        x = apply_conditioning(x, cond_noised, self.action_dim) 

        # Now we denoise it again
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_constrained(self, x, global_cond, cond, t, constraint_scale = 30):
        b, *_, device = *x.shape, x.device
        noise= torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)

        model_mean = model_mean.clone().detach()
        model_mean.requires_grad_()
        optimizer = torch.optim.Adam([model_mean], lr=1e-2)
        losses = []
        for i in range(8):
            optimizer.zero_grad()
            l = 7*mountain_loss(model_mean) + 50 * smoothness_loss(model_mean)
            # print(d)
            losses.append(l.item())
            l.backward()
            optimizer.step()
            # model_mean = apply_conditioning(model_mean, cond, self.action_dim)

        # adjust_mean = self.compute_constraint_gradient(x, cond) * constraint_scale
        # adjust_mean = adjust_mean * constraint_scale * (nonzero_mask * (0.5 * model_log_variance).exp())

        # try alternating between sampling from the model and sampling from the constraint
        # if i % 2 != 0:
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise    


    @torch.no_grad()
    def move_towards_constraint(self, x, global_cond, cond, t, constraint_scale=15):
        b, *_, device = *x.shape, x.device
        noise= torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
        adjust_mean = self.compute_constraint_gradient(x, cond) 
        # adjust_mean = adjust_mean * constraint_scale * (nonzero_mask * (0.5 * model_log_variance).exp())
        adjust_mean = adjust_mean * constraint_scale
        # return x + adjust_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x + adjust_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_original(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, global_cond, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    # @torch.no_grad()
    def p_sample_loop_constrained(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample_constrained(x, global_cond, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def p_sample_loop_estimate_feature(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        a_probs = []
        b_probs = []
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, a_given_x, b_given_x = self.p_sample_estimate_feature(x, global_cond, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            a_probs.append(a_given_x)
            b_probs.append(b_given_x)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        a_probs = torch.stack(a_probs, dim=1).sum(dim=1)
        b_probs = torch.stack(b_probs, dim=1).sum(dim=1)

        print(a_probs)
        print(b_probs)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def p_sample_loop_repaint(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
        for sample in self.p_sample_loop_progressive(shape, global_cond, cond, verbose, return_diffusion):
            final = sample
        return final

    def p_sample_loop_cfm(self, shape, global_cond, verbose=True, return_diffusion=False):
        # traj = self.node.trajectory(
        #     torch.randn(shape).to(self.device),
        #     t_span=torch.linspace(0, 1, 100).to(self.device), # number of denoising timesteps is 50
        #     global_cond = global_cond
        # )
        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(t, x, global_cond=global_cond),
            torch.randn(shape).to(self.device),
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )

        # traj = torchdiffeq.odeint(
        #     lambda t, x: self.model.forward(t, x, global_cond=global_cond),
        #     torch.randn(shape).to(self.device),
        #     torch.linspace(0, 1, 2).to(self.device),
        #     atol=1e-4,
        #     rtol=1e-4,
        #     method="euler",
        # )

        return traj[-1]

    # @torch.no_grad()
    def p_sample_loop(self, shape, global_cond, cond, verbose=True, return_diffusion=False, **kwargs):
        sample_type = kwargs.get('sample_type', 'original')

        return self.p_sample_loop_cfm(shape, global_cond, verbose, return_diffusion)
    

        if sample_type == 'repaint':
            return self.p_sample_loop_repaint(shape, global_cond, verbose, return_diffusion)
        elif sample_type == 'constrained':
            return self.p_sample_loop_constrained(shape, global_cond, verbose, return_diffusion)
        elif sample_type == 'original':
            return self.p_sample_loop_original(shape, global_cond, verbose, return_diffusion)
        elif sample_type == 'estimate_feature':
            return self.p_sample_loop_estimate_feature(shape, global_cond, verbose, return_diffusion)
        else:
            raise NotImplementedError

    def undo(self, img_after_model, t, debug=False):
        return self._undo(img_after_model, t)    

    def _undo(self, x, t, device="cuda"):
        # need to make this correct
        # beta = _extract_into_tensor(self.betas, t, x.shape)
        # x_noisy = torch.sqrt(1-beta) * x + torch.sqrt(beta) * torch.randn_like(x)
        # return x_noisy
    
        # self.register_buffer('sqrt_betas', torch.sqrt(betas))
        sqrt_betas = torch.sqrt(self.betas)
        sqrt_one_minus_betas = torch.sqrt(1 - self.betas)

        t_full = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        noise = torch.randn_like(x).to(device)
        x_noisy =  (
            extract(sqrt_one_minus_betas, t_full, x.shape) * x +
            extract(sqrt_betas, t_full, x.shape) * noise
        )
        return x_noisy


    def compute_smoothness(self, traj):
        """ Currently not used on REPAINT branch !!! """
        smoothness_out = (traj[:, :-1, :] - traj[:, 1:, :])**2
        smoothness_out = torch.sqrt(smoothness_out.sum(dim=-1))

        return smoothness_out.mean()

    def compute_constraint_gradient(self, traj, cond):
        """ Compute the gradient to make the constraint satisfied.
         Currently not used on REPAINT branch !!! 
        One issue is how to ignore the gradient in areas of -1?

        """
        assert cond is not None
        with torch.enable_grad():
            traj = traj.detach().requires_grad_(True)
            out = self.compute_smoothness(traj)
            ret_val = torch.autograd.grad(out, traj)[0]
        
        # for all the indices in cond, set the gradient to 0
        for b, c in enumerate(cond):
            ret_val[b, c[0]] = 0 

        return ret_val

    # @torch.no_grad()
    def conditional_sample(self, global_cond, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.device
        batch_size = len(cond)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        # global_cond = global_cond.to(device)
        # global_cond = {k: v.to(device) for k, v in global_cond.items()}
        for k, v in global_cond.items():
            if type(v) is torch.Tensor:
                global_cond[k] = v.to(device)

        return self.p_sample_loop(shape, global_cond, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, global_cond, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, t, global_cond=global_cond)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    @property
    def device(self):
        """Get the device where the model's parameters are allocated."""
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device

    def loss(self, x, global_cond, cond):
        # cond is for applying constraints but currently not implemented in this branch
        x = x.to(self.device)
        for k, v in global_cond.items():
            if type(v) is torch.Tensor:
                global_cond[k] = v.to(self.device)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        vt = self.model(t, xt, global_cond=global_cond)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {'loss': loss.item()}

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

