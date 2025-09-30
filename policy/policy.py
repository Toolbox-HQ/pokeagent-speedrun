from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def get_action(self) -> list:
        pass

    @abstractmethod
    def send_state(self, state):
        pass

    def __next__(self):
        return self.get_action()
    
# This is the format for MGBA emulator
MGBA_KEY_LIST = [
    ["a"],
    ["b"],
    ["start"],
    ["select"],
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    [],
]

# This is the format we use for logging / training
KEY_LIST_FOR_TRAINING = [
    "a",
    "b",
    "start",
    "select",
    "up",
    "down",
    "left",
    "right",
    "none",
]

CLASS_TO_KEY = {ind:key for (ind, key) in enumerate(KEY_LIST_FOR_TRAINING)}
KEY_TO_CLASS = {v:k for k,v in CLASS_TO_KEY.items()}