from gym import Env
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Literal
from IPython.display import display, clear_output

from .agent import Agent
from .config import Config
from .evaluator import Evaluator
from ..shared.utils import (
    get_output_folder,
    plot_rays,
    show_img,
)


class Runner:
    def __init__(
            self,
            train_loader,
            test_loader,
            env: Env,
            config: Config,
            save_path=None,
            evaluate=False,
            tensorboard=False,
            with_img=False,
            plot_rays=False,
            num_rays=1,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.env = env

        self.episodes = config.episodes
        self.episode_steps = config.episode_steps
        self.max_episodes = config.max_episodes
        self.eval_episodes = config.eval_episodes
        self.warmup = config.warmup
        self.save_path = get_output_folder(config.save_path) \
            if save_path is None else save_path

        self.evaluate = evaluate
        self.tensorboard = tensorboard
        self.writer = SummaryWriter(log_dir=self.save_path) if tensorboard else None
        self.with_img = with_img
        self.plot_rays = plot_rays
        self.num_rays = num_rays

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.agent = Agent(self.state_dim, self.action_dim, config, env)
        self.evaluator = Evaluator(
            self.env,
            self.eval_episodes,
            config.eval_episode_steps,
            self.writer,
            tensorboard,
            self.save_path,
        )

    def __call__(
            self,
            mode: Literal['train', 'test'],
            load_weights=None
    ):
        loader, run = self.route(mode)
        try:
            if load_weights:
                self.agent.load_weights(
                    load_weights['actor'],
                    load_weights['critic']
                )
            for idx, rb in enumerate(loader, start=1):
                self.env.initial_state = rb
                self.agent.initial_state = self.env.initial_state
                self.agent.clip_angle = rb.clip_angle
                self.agent.batch_size = rb.size
                self.agent.chunk_size = rb.chunk_size
                self.agent.imshape = rb.imshape
                self.agent.imsize = rb.imsize
                self.agent.camera_intrinsics = rb.camera_intrinsics
                self.agent.forget()
                run(idx, rb)
        except KeyboardInterrupt:
            pass
        if self.tensorboard:
            self.writer.close()

    def route(self, mode: Literal['train', 'test']):
        if mode == 'train':
            return self.train_loader, self.train
        elif mode == 'test':
            return self.test_loader, self.test
        else:
            raise RuntimeError(f'invalid mode: {mode}')

    def train(self, batch_idx, batch):
        self.agent.is_training = True # TODO: remove?
        train_step = 0

        if self.plot_rays or self.with_img:
            fig, ax = None, None

        episodes = range(1, self.episodes + 1) if self.max_episodes \
            else itertools.count(1)

        for episode in episodes:
            # reset at start of episode
            state = self.env.reset()

            ep_reward = torch.zeros(1, device=state.device)
            ep_q1_loss, ep_q2_loss, ep_policy_loss = 0., 0., 0.

            time_to_save = episode % 10 == 0

            for step in range(1, self.episode_steps + 1):
                log = f'[train] batch {batch_idx} episode {episode} step {step}'
                if not (time_to_save and self.with_img):
                    display(log)
                    clear_output(wait=True)

                with torch.no_grad():
                    # agent picks an action
                    if train_step < self.warmup:
                        action = self.agent.random_action(state)
                    else:
                        action, _ = self.agent.select_action(state)
                    # env response
                    next_state, reward, done, info = self.env.step(action, self.with_img)
                # agent observes next state - updates policy
                self.agent.observe(state, action, reward, done)
                if train_step >= self.warmup:
                    policy_loss, q1_loss, q2_loss = self.agent.update_policy()
                # plot action
                if self.plot_rays:
                    fig, ax = plot_rays(
                        next_state[:batch.imsize, 3:],
                        batch.imsize // self.num_rays,
                        fig, ax,
                        title=log,
                    )
                # update plots
                if time_to_save:
                    if self.with_img:
                        fig, ax = show_img(info['img'], fig, ax, stdout=True, title=log)
                    if self.tensorboard:
                        self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/mean_reward', reward.mean().item(), step)
                        self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/done_count', info['done_count'], step)
                        self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/done_ratio', info['done_ratio'], step)
                        if train_step >= self.warmup:
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/policy_loss', policy_loss, step)
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/q1_loss', q1_loss, step)
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/q2_loss', q2_loss, step)
                        self.writer.flush()
                # update
                state = next_state
                train_step += 1
                ep_reward = ep_reward + reward
                if train_step - 1 >= self.warmup:
                    ep_q1_loss += q1_loss
                    ep_q2_loss += q2_loss
                    ep_policy_loss += policy_loss
                # end of episode
                if done:
                    self.agent.observe_done(state)
                    break
            # save model and img
            if time_to_save:
                self.agent.save_model(self.save_path)
                fig, ax = show_img(info['img'], fig, ax, write_path=self.save_path,
                                   title=f'[train] batch {batch_idx} episode {episode} step {step}')
            # track episode reward
            if self.tensorboard:
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_mean_reward', ep_reward.mean().item(), episode)
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_steps', step, episode)
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_done_count', info['done_count'], episode)
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_done_ratio', info['done_ratio'], episode)
                if train_step - 1 >= self.warmup:
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_policy_loss', ep_policy_loss / step, episode)
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_q1_loss', ep_q1_loss / step, episode)
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_q2_loss', ep_q2_loss / step, episode)
                self.writer.flush()

    def test(self, batch_idx, _):
        self.agent.is_training = False # TODO: remove?
        self.agent.eval()

        policy = lambda s: self.agent.select_action(s)
        self.evaluator(policy, batch_idx)
