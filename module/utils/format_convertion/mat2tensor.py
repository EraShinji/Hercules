from scipy import io
from torch import tensor
from module.utils.format_convertion.convertion import Convertion
from typing import Dict, List, Optional
import wfdb

class Mat2Tensor(Convertion):
    """
    A class to load ECG data from .mat/.hea file pairs and convert it to a PyTorch Tensor.
    """
    def __init__(self, basic_path: str, lead_name: str = "I"):
        """
        Initializes the Mat2Tensor converter.

        Args:
            basic_path (str): The base path of the record, without the file extension.
            lead_name (str): Name of the lead to extract (e.g., "I", "II", "V1", etc.)
        """
        super().__init__(basic_path)
        self.dat_path = basic_path + ".mat"
        self.hea_path = basic_path + ".hea"
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
        Load ECG data from .mat and .hea files.
        .mat format:MATLAB ECG data
        .hea format: Header file containing metadata
        """
        # 1. Load MAT file (usually contains signal arrays)
        # loadmat returns a dictionary, usually signals are stored in 'val' or 'data' keys
        mat_data = io.loadmat(self.dat_path)
        full_data = mat_data.get('val') if 'val' in mat_data else list(mat_data.values())[-1]

        # 2. Parse hea file to extract lead info and comments
        self._parse_hea()

        # 3. Extract specified lead data
        # MAT file shape is (leads, samples), need row slicing
        if self.lead_index is not None and full_data is not None:
            self.dat = full_data[self.lead_index, :]
        else:
            self.dat = full_data

        return self.dat, self.hea

    def _parse_hea(self):
        """
        Parse .hea header file using wfdb and set lead_index based on lead_name
        """
        record = wfdb.rdheader(self.basic_path)
        self.hea = record.__dict__

        # Get lead name list and find the index of target lead
        if hasattr(record, 'sig_name') and record.sig_name:
            try:
                self.lead_index = record.sig_name.index(self.lead_name)
            except ValueError:
                raise ValueError(f"Lead name '{self.lead_name}' not found in file. Available leads: {record.sig_name}")
    def to_tensor(self):
        if self.dat is None:
            raise ValueError("Data not loaded.")
        self.dat_tensor = tensor(self.dat)
        return self.dat_tensor

if __name__ == '__main__':
    # Example usage:
    from pathlib import Path
    io.loadmat("/Users/aleclanned/PycharmProjects/Hercules/data/for_test/mat/JS00001.mat")
    project_root = Path(__file__).resolve().parents[4]
    base_path = str(project_root / "Hercules"/ "data" / "for_test" / "mat" / "JS00001")

    # Extract lead I data
    converter = Mat2Tensor(base_path, lead_name="I")
    print("Lead name:", converter.lead_name)
    print("Lead index:", converter.lead_index)
    print("HEA comments:", converter.hea)
    print("Data tensor shape:", converter.dat_tensor.shape)