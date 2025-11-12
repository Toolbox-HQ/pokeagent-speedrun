from abc import ABC, abstractmethod
from models.dataclass import PolicyConfig

class Policy(ABC):

    def __init__(self, cfg: PolicyConfig):
        self.config: PolicyConfig = cfg        

    @abstractmethod
    def get_action(self) -> list:
        pass

    @abstractmethod
    def send_state(self, state):
        pass

    def __next__(self):
        return self.get_action()