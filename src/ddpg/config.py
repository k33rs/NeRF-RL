from dataclasses import dataclass, field
import torch


@dataclass
class Config():
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    net: dict = field(default_factory=lambda: {
        'hidden1': 400,
        'hidden2': 300,
        'init_w': 3e-3,
    })
    save_path: str = 'out'
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    mem_batch_size: int = 64
    gamma: float = 0.99 # discount factor
    tau: float = 1e-3 # soft target update factor
    eps: float = 1.
    eps_decay: float = 2e-5
    episodes: int = 400
    episode_steps: int = 500
    warmup: int = 1000
    max_steps: bool = True
    replay_bufsize: int = 1_000_000
    window_length: int = 1
    eval_episodes: int = 10
    eval_episode_steps: int = 500 # max length of episode in eval mode
