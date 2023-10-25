from gym import Env
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Literal
from IPython.display import display, clear_output

from .agent import Agent
from .config import Config
from .evaluator import Evaluator
from ..shared.utils import get_output_folder, show_img


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
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.env = env

        self.episodes = config.episodes
        self.episode_steps = config.episode_steps
        self.num_steps = self.episodes * self.episode_steps
        self.max_steps = config.max_steps
        self.eval_episodes = config.eval_episodes
        self.warmup = config.warmup
        self.save_path = get_output_folder(config.save_path) \
            if save_path is None else save_path

        self.evaluate = evaluate
        self.tensorboard = tensorboard
        self.writer = SummaryWriter(log_dir=self.save_path) if tensorboard else None
        self.with_img = with_img

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
            load_weights=None,
            with_penalty=False,
    ):
        loader, run = self.route(mode)
        results = None # stat test results
        try:
            if load_weights:
                self.agent.load_weights(
                    load_weights['actor'],
                    load_weights['critic']
                )
            for idx, rb in enumerate(loader, start=1):
                self.env.initial_state = rb
                self.env.init_img()
                self.agent.initial_state = self.env.initial_state
                self.agent.clip_angle = rb.clip_angle
                self.agent.clip_angle_small = rb.clip_angle_small
                self.agent.batch_size = rb.size
                self.agent.imshape = rb.imshape
                self.agent.imsize = rb.imsize
                self.agent.camera_intrinsics = rb.camera_intrinsics
                self.agent.forget()
                res = run(idx, with_penalty)
                if res is not None:
                    if results is None:
                        results = []
                    results.append(res)
        except KeyboardInterrupt:
            loader.reset()
        if self.tensorboard:
            self.writer.close()
        return results

    def route(self, mode: Literal['train', 'test']):
        if mode == 'train':
            return self.train_loader, self.train
        elif mode == 'test':
            return self.test_loader, self.test

    def train(self, batch_idx, with_penalty):
        self.env.is_training = True
        self.env.with_penalty = with_penalty
        train_step = 1
        stop = False

        if self.with_img:
            fig, ax = None, None

        for episode in itertools.count(1):
            # reset at start of episode
            state = self.env.reset()

            ep_reward = torch.zeros(1, device=state.device)
            ep_q1_loss, ep_q2_loss, ep_policy_loss = 0., 0., 0.

            time_to_save = episode % 10 == 0

            for step in range(1, self.episode_steps + 1):
                log = f'[train step {train_step}] batch {batch_idx} episode {episode} step {step}'
                if not (time_to_save and self.with_img):
                    display(log)
                    clear_output(wait=True)

                with torch.no_grad():
                    # agent picks an action
                    if train_step <= self.warmup:
                        action = self.agent.random_action(state)
                    else:
                        action, _ = self.agent.select_action(state)
                    # env response
                    next_state, reward, done, info = self.env.step(action)
                # max number of steps
                if step == self.episode_steps:
                    done = True
                # agent observes next state - updates policy
                self.agent.observe(state, action, reward, done)
                if train_step > self.warmup:
                    policy_loss, q1_loss, q2_loss = self.agent.update_policy()
                    ep_q1_loss += q1_loss
                    ep_q2_loss += q2_loss
                    ep_policy_loss += policy_loss
                # update plots
                if time_to_save:
                    if self.with_img and train_step > self.warmup:
                        fig, ax = show_img(info['img_coords'], fig, ax, stdout=True, title=log)
                    if self.tensorboard:
                        self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/mean_reward', reward.mean().item(), step)
                        self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/done', info['done'], step)
                        if train_step > self.warmup:
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/policy_loss', policy_loss, step)
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/q1_loss', q1_loss, step)
                            self.writer.add_scalar(f'[train]_b{batch_idx}_ep{episode}/q2_loss', q2_loss, step)
                        self.writer.flush()
                # update
                state = next_state
                train_step += 1
                ep_reward = ep_reward + reward
                # end of training
                if self.max_steps and train_step >= self.num_steps:
                    stop = True
                # end of episode
                if done:
                    self.agent.observe_done(state)
                    break
            # save model and img
            if time_to_save:
                self.agent.save_model(self.save_path)
                fig, ax = show_img(info['img_coords'], fig, ax, write_path=self.save_path, title=log)
            # track episode reward
            if self.tensorboard:
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_mean_reward', ep_reward.mean().item(), episode)
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_steps', step, episode)
                self.writer.add_scalar(f'[train]_b{batch_idx}/ep_done', info['done'], episode)
                if train_step - 1 > self.warmup:
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_policy_loss', ep_policy_loss / step, episode)
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_q1_loss', ep_q1_loss / step, episode)
                    self.writer.add_scalar(f'[train]_b{batch_idx}/ep_q2_loss', ep_q2_loss / step, episode)
                self.writer.flush()
            # end of training
            if stop:
                break

    def test(self, batch_idx, with_penalty):
        self.agent.eval()
        self.env.is_training = False
        self.env.with_penalty = with_penalty

        policy = lambda s: self.agent.select_action(s)
        results = self.evaluator(policy, batch_idx)
        return results
