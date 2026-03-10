from dataclasses import dataclass
from module.utils.format_convertion import convertion
import wfdb
import numpy as np

from module.utils.format_convertion.convertion import Convertion


class Dat2Tensor(Convertion):
    def __init__(self,  basic_path: str, lead_name: str = "I"):
        super().__init__(basic_path)
        self.dat = None
        self.hea = None
        self.dat_tensor = None
        self.lead_name = lead_name
        self.lead_index = None
        self.hea_comments = {}
        self._load()
        self.to_tensor()

    def _load(self):
        """
        Load .dat signal data and .hea header file data
        """
        # Use wfdb to read dat/hea file pair
        record = wfdb.rdrecord(self.basic_path)
        full_data = record.p_signal
        self._parse_hea()
        if self.lead_index is not None and full_data is not None:
            self.dat = full_data[:, self.lead_index]
        else:
            self.dat = full_data

        return self.dat, self.hea
    def _parse_hea(self):
        """
        Parse .hea header file data and set lead_index based on lead_name
        """
        record = wfdb.rdheader(self.basic_path)
        self.hea = record.__dict__

        # Get lead name list and find the index of target lead
        if hasattr(record, 'sig_name') and record.sig_name:
            try:
                self.lead_index = record.sig_name.index(self.lead_name)
            except ValueError:
                raise ValueError(f"Lead name '{self.lead_name}' not found in file. Available leads: {record.sig_name}")

        return self.hea


    def to_tensor(self):
        """
        Convert loaded data to PyTorch Tensor
        """
        if self.dat is None:
            raise ValueError("Data not loaded.")
        from torch import tensor
        self.dat_tensor = tensor(self.dat)
        return self.dat_tensor
if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[4]
    base_path = str(project_root / "Hercules"/ "data" / "for_test" / "dat" / "103554663")

    dat2tensor = Dat2Tensor(base_path)
    print(dat2tensor.hea)
    print(dat2tensor.dat_tensor)
    print("Data tensor shape:", dat2tensor.dat_tensor.shape)
