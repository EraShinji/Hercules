from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from torch import Tensor

class Convertion(ABC):
    basic_path:str
    dat: Optional[np.ndarray]
    hea: Optional[List[str]]
    dat_tensor: Optional[Tensor]

    @abstractmethod
    def to_tensor(self):
        pass
    @abstractmethod
    def load(self):
        pass