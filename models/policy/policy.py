from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def get_action(self) -> list:
        pass

    def __next__(self):
        return self.get_action()
