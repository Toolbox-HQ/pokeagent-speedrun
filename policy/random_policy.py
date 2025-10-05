from policy.policy import Policy, KEY_TO_MGBA
import random
from dataclass import PolicyConfig

class RandomPolicy(Policy):

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(cfg)
        self.key_map = KEY_TO_MGBA.copy()

        for key in cfg.exclude:
            del self.key_map[key]

        self.action_queue = []

    def enqueue_action(self) -> None:
        num_presses = random.randint(150, 300)
        button = random.choice(list(self.key_map.values()))
        self.action_queue.extend(button * num_presses)

    def get_action(self)-> list:

        while not self.action_queue:
            self.enqueue_action()

        return [self.action_queue.pop(0)]
    
    def send_state():
        pass

class RandomMovementPolicy(RandomPolicy):

    def enqueue_action(self) -> None:

        # 150-300 frame actions presses ~ 3-5 seconds
        num_presses = random.randint(150, 300)

        MGBA_MOVEMENT_KEY_LIST = [
            ["up"],
            ["down"],
            ["left"],
            ["right"],
        ]
        button = random.choices(MGBA_MOVEMENT_KEY_LIST, weights=[1,1,1,1])[0]
        self.action_queue.extend(button * num_presses)