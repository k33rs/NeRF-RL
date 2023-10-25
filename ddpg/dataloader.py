from typing import Callable


class CustomDL:
    def __init__(
            self,
            next_call: Callable,
            cameras = None,
            step_size = 1,
    ):
        self.next_call = next_call
        self.cameras = cameras
        self.step = 0
        self.step_size = step_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= len(self.cameras):
            self.reset()
            raise StopIteration
        batch = self[self.step]
        self.step += self.step_size
        return batch
    
    def __getitem__(self, step):
        return self.next_call(self.cameras, range(step, step+self.step_size))
    
    def __len__(self):
        return len(self.cameras)
    
    def reset(self, step=0):
        self.step = step
