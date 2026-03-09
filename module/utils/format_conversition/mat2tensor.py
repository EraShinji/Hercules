from scipy import io
from torch import tensor
from module.utils.format_conversition.convertion import Convertion

class Mat2Tensor(Convertion):
    """
    A class to load ECG data from .mat/.hea file pairs and convert it to a PyTorch Tensor.
    """
    def __init__(self, basic_path: str):
        """
        Initializes the Mat2Tensor converter.

        Args:
            basic_path (str): The base path of the record, without the file extension.
        """
        self.basic_path = basic_path
        self.dat_path = self.basic_path + ".mat"
        self.hea_path = self.basic_path + ".hea"
        self.dat = None
        self.hea = None
        self.dat_tensor = None
        self._load()



    def _load(self):
        """
        加载 .mat 信号数据和 .hea 文本头文件数据
        """
        # 1. 加载 MAT 文件 (通常包含信号数组)
        # loadmat 返回一个字典，通常信号存储在 'val' 或 'data' 键下
        mat_data = io.loadmat(self.dat_path)
        self.dat = mat_data.get('val') if 'val' in mat_data else list(mat_data.values())[-1]

        # 2. 读取 .hea 文本数据
        try:
            with open(self.hea_path, 'r', encoding='utf-8') as f:
                self.hea = f.readlines()
        except FileNotFoundError:
            print(f"警告: 未找到头文件 {self.hea_path}")
            self.hea = None

        return self.dat, self.hea
    def to_tensor(self):
        if self.dat is None:
            raise ValueError("Data not loaded. Please call .load() before .to_tensor().")
        self.dat_tensor = tensor(self.dat)

if __name__ == '__main__':
    # Example usage:
    converter = Mat2Tensor("C:/Users/aleclanned/PycharmProjects/Hercules/data/for_test/mat/JS00001")
    signal_tensor = converter.to_tensor()
    print(signal_tensor.dat_tensor)