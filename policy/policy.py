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
    
KEY_LIST = [
    ["a"],
    ["b"],
    ["start"],
    ["select"],
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    ["l"],
    ["r"],
    [],
]