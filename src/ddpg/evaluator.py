
from gym import Env
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
from ..shared.utils import show_img
from ..shared.geom import camera_to_image


class Evaluator:
    def __init__(
            self,
            env: Env,
            episodes,
            episode_steps,
            writer: SummaryWriter,
            tensorboard=False,
            save_path=None,
    ):
        self.env = env
        self.episodes = episodes
        self.episode_steps = episode_steps
        self.writer = writer
        self.tensorboard = tensorboard
        self.with_img = save_path is not None
        self.save_path = save_path

    def __call__(self, policy, batch_idx):
        if self.with_img:
            fig, ax = None, None
            fig3d, ax3d = None, None

        for episode in range(1, self.episodes + 1):
            # reset at the start of episode
            state = self.env.reset()
            ep_reward = torch.zeros(1, device=state.device)

            _, img = self.env.eval_model()

            # start episode
            for step in range(1, self.episode_steps + 1):
                log = f'[eval] batch {batch_idx} episode {episode} step {step}'
                if not self.with_img:
                    display(log)
                    clear_output(wait=True)
                # take action, collect reward
                with torch.no_grad():
                    action = policy(state)
                    state, reward, done, info = self.env.step(action)
                # update plots
                if self.with_img:
                    two_img = np.concatenate((img, info['img']), axis=0)
                    fig, ax = show_img(two_img, fig, ax, title=log, stdout=True)
                if self.tensorboard:
                    self.writer.add_scalar(f'[eval]_b{batch_idx}_ep{episode}/mean_reward', reward.mean().item(), step)
                    self.writer.add_scalar(f'[eval]_b{batch_idx}_ep{episode}/done', info['done'], step)
                # update
                ep_reward = ep_reward + reward
                # end of episode
                if done:
                    break

            if self.with_img:
                two_img = np.concatenate((img, info['img']), axis=0)
                fig, ax = show_img(two_img, fig, ax, title=log, write_path=self.save_path)
                fig3d, ax3d = show_img(two_img, fig3d, ax3d, title=f'3D {log}', write_path=self.save_path, three_d=True)
            if self.tensorboard:
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_mean_reward', ep_reward.mean().item(), episode)
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_steps', step, episode)
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_done', info['done'], episode)
                self.writer.flush()

        # IS integral estimator
        if not self.env.is_training:
            f_values = self.env.img
            density = self.env.get_density()
            intgr = (f_values / density).mean()
        
            if self.with_img:
                desired = self.env.get_density(desired=True)
                two_img = np.concatenate((density, desired), axis=0)
                show_img(two_img, fig3d, ax3d, title=f'density {log}', write_path=self.save_path, three_d=True)

            return intgr
