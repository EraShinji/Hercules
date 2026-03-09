from dataclasses import dataclass
from module.utils.format_conversition import convertion
import wfdb
import numpy as np

from module.utils.format_conversition.convertion import Convertion


@dataclass
class Dat2Tensor(Convertion):
    def __init__(self, dat_path:str, hea_path:str):
        self.dat_path = dat_path
        self.hea_path = hea_path
        self.dat = None
        self.hea = None
        self.dat_tensor = None

    def load(self):
        pass #TODO

