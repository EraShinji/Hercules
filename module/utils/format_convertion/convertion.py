from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from torch import Tensor

class Convertion(ABC):
    dat: Optional[np.ndarray]
    hea: Optional[List[str]]
    dat_tensor: Optional[Tensor]

    def __init__(self, basic_path: str):
        self.basic_path = basic_path

    @abstractmethod
    def to_tensor(self):
        pass

    @abstractmethod
    def _load(self):
        pass
    @abstractmethod
    def _parse_hea(self):
        pass