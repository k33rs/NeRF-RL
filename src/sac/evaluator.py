
from gym import Env
import torch
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
from ..shared.utils import show_img


class Evaluator(object):
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

        for episode in range(1, self.episodes + 1):
            # reset at the start of episode
            state = self.env.reset()
            ep_reward = torch.zeros(1, device=state.device)

            # start episode
            for step in range(1, self.episode_steps + 1):
                log = f'[eval] batch {batch_idx} episode {episode} step {step}'
                if not self.with_img:
                    display(log)
                    clear_output(wait=True)
                # take action, collect reward
                with torch.no_grad():
                    action, _ = policy(state)
                    state, reward, done, info = self.env.step(action)
                # update plots
                if self.with_img:
                    fig, ax = show_img(info['img_coords'], fig, ax, title=log, stdout=True)
                if self.tensorboard:
                    self.writer.add_scalar(f'[eval]_b{batch_idx}_ep{episode}/mean_reward', reward.mean().item(), step)
                    self.writer.add_scalar(f'[eval]_b{batch_idx}_ep{episode}/done', info['done'], step)
                # update
                ep_reward = ep_reward + reward
                # end of episode
                if done:
                    break

            if self.with_img:
                fig, ax = show_img(info['img_coords'], fig, ax, title=log, write_path=self.save_path)
            if self.tensorboard:
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_mean_reward', ep_reward.mean().item(), episode)
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_steps', step, episode)
                self.writer.add_scalar(f'[eval]_b{batch_idx}/ep_done', info['done'], episode)
                self.writer.flush()

        # IS integral estimator
        f_values = self.env.img
        density = self.env.get_density()
        non_zero = density > 0
        intgr = (f_values[non_zero] / density[non_zero]).mean()

        return intgr
