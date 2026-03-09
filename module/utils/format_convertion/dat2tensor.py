from dataclasses import dataclass
from module.utils.format_convertion import convertion
import wfdb
import numpy as np

from module.utils.format_convertion.convertion import Convertion


@dataclass
class Dat2Tensor(Convertion):
    def __init__(self, dat_path:str, hea_path:str):
        self.dat_path = dat_path
        self.hea_path = hea_path
        self.dat = None
        self.hea = None
        self.dat_tensor = None
        self._load()
        self.to_tensor()

    def _load(self):
        """
        加载 .dat 信号数据和 .hea 头文件数据
        """
        # 使用 wfdb 读取 dat/hea 文件对
        record = wfdb.rdrecord(self.dat_path.replace('.dat', ''))
        self.dat = record.p_signal
        self.hea = record.__dict__
        return self.dat, self.hea

    def to_tensor(self):
        """
        将加载的数据转换为 PyTorch Tensor
        """
        if self.dat is None:
            raise ValueError("Data not loaded.")
        from torch import tensor
        self.dat_tensor = tensor(self.dat)
        return self.dat_tensor

