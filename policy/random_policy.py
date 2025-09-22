from policy.policy import Policy, KEY_LIST
import numpy as np
import random

class RandomPolicy(Policy):

    def __init__(self) -> None:
        self.game_fps = 60
        self.action_queue = []
        self.mean_action_time = 1.0
        self.std_action_time = 0.25

    def enqueue_action(self) -> None:
        rnd = np.random.normal(loc=self.mean_action_time,
                               scale=self.std_action_time,
                               size=1).item()
        
        num_presses = round(rnd * self.game_fps)
        button = random.choice(KEY_LIST)
        self.action_queue.extend([button] * num_presses)

    def get_action(self)-> list:

        while not self.action_queue:
            self.enqueue_action()

        return [self.action_queue.pop(0)]
    
    def send_state():
        pass