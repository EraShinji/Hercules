from scipy import io
from torch import tensor
from module.utils.format_convertion.convertion import Convertion
from typing import Dict, List, Optional

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
        self.basic_path = basic_path
        self.dat_path = self.basic_path + ".mat"
        self.hea_path = self.basic_path + ".hea"
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

        # 2. Read .hea text data
        try:
            with open(self.hea_path, 'r', encoding='utf-8') as f:
                self.hea = f.readlines()
        except FileNotFoundError:
            print(f"Failed to load header file {self.hea_path}")
            self.hea = None
            self.dat = full_data
            return self.dat, self.hea

        # 3. Parse hea file to extract lead info and comments
        self._parse_hea()

        # 4. Extract specified lead data
        if self.lead_index is not None and full_data is not None:
            self.dat = full_data[self.lead_index]
        else:
            self.dat = full_data

        return self.dat, self.hea

    def _parse_hea(self):
        """
        Parse hea file content to extract lead indices and comments.
        """
        if not self.hea:
            return

        lead_names = []
        comments = {}

        for line in self.hea:
            line = line.strip()
            if not line:
                continue

            # Parse comment lines (starting with #)
            if line.startswith('#'):
                # Extract key-value pairs like #Age: 85
                if ':' in line:
                    key_value = line[1:].split(':', 1)  # Remove # and split
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        comments[key] = value
                continue

            # Parse lead info lines (format: filename.mat ... lead_name)
            parts = line.split()
            if len(parts) >= 3 and parts[0].endswith('.mat'):
                # Last part is the lead name
                lead_name = parts[-1]
                lead_names.append(lead_name)

        self.hea_comments = comments

        # Find the index of the requested lead
        if self.lead_name in lead_names:
            self.lead_index = lead_names.index(self.lead_name)
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
    print("HEA comments:", converter.hea_comments)
    print("Data tensor shape:", converter.dat_tensor.shape)