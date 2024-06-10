from dataclasses import dataclass
import torch


@dataclass
class Config():
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path: str = 'out'
    lr: float = 3e-4
    mem_batch_size: int = 64
    alpha: float = 0.2
    gamma: float = 0.99 # discount factor
    tau: float = 5e-3 # soft target update factor
    episodes: int = 200
    episode_steps: int = 1000
    warmup: int = 1000
    max_episodes: bool = True
    replay_bufsize: int = 6000000
    window_length: int = 1
    eval_episodes: int = 10
    eval_episode_steps: int = 1000 # max length of episode in eval mode
    action_samples: int = 360
