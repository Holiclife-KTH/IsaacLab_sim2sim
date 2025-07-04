from abc import ABC, abstractmethod
from omni.isaac.core.utils.types import ArticulationAction

class BaseController(ABC):
    """[summary]
    
    Args:
        name (str): [description]
    """

    def __init__(self, name: str) -> None:
        self._name = name


    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """A controller should take inputs and returns an ArticulationAction to be then passed to the
           ArticulationController.

        Args:
            observations (dict): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            ArticulationAction: [description]
        """
        raise NotImplementedError
    

    def reset(self) -> None:
        """REsets state of the controller"""
        return
    


