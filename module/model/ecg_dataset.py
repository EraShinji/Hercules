from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from module.utils.format_convertion.mat2tensor import Mat2Tensor
from module.utils.format_convertion.dat2tensor import Dat2Tensor


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG data loading from MAT or DAT format files.
    
    Supports loading single lead ECG signals for training deep learning models.
    """
    
    def __init__(
        self,
        data_paths: List[Union[str, Path]],
        lead_name: str = "I",
        file_format: Optional[str] = None,  # 'mat', 'dat', or None (auto-detect)
        labels: Optional[List[int]] = None,
        transform: Optional[callable] = None,
        target_length: Optional[int] = None,  # Fixed length for padding/truncation
    ):
        """
        Initialize ECG Dataset.
        
        Args:
            data_paths: List of file paths (without extension) to ECG records
            lead_name: Name of the ECG lead to extract (default: "I")
            file_format: File format ('mat' or 'dat'), auto-detect if None
            labels: Optional list of labels for supervised learning
            transform: Optional transform function to apply to the data
            target_length: Target sequence length (pad or truncate to this length)
        """
        self.data_paths = [Path(p) for p in data_paths]
        self.lead_name = lead_name
        self.file_format = file_format
        self.labels = labels
        self.transform = transform
        self.target_length = target_length
        
        # Validate labels length matches data paths
        if labels is not None and len(labels) != len(data_paths):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of data paths ({len(data_paths)})")
    
    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format based on existing files."""
        if (path.parent / f"{path.name}.mat").exists():
            return "mat"
        elif (path.parent / f"{path.name}.dat").exists():
            return "dat"
        else:
            raise FileNotFoundError(f"No MAT or DAT file found for: {path}")
    
    def _load_record(self, path: Path) -> torch.Tensor:
        """Load a single ECG record and return as tensor."""
        fmt = self.file_format or self._detect_format(path)
        
        if fmt == "mat":
            converter = Mat2Tensor(str(path), lead_name=self.lead_name)
        elif fmt == "dat":
            converter = Dat2Tensor(str(path), lead_name=self.lead_name)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")
        
        return converter.dat_tensor
    
    def _pad_or_truncate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad or truncate tensor to target length."""
        if self.target_length is None:
            return tensor
        
        current_length = tensor.shape[0]
        
        if current_length > self.target_length:
            # Truncate
            return tensor[:self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            padding = torch.zeros(self.target_length - current_length)
            return torch.cat([tensor, padding])
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        # WARNING DIY Dataset must implement __getitem__ method.
        # Must read data and return tensor(s) which be processed
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

        """
        Get a single ECG record.
        Returns:
            If labels provided: (tensor, label)
            If no labels: tensor
        """
        path = self.data_paths[idx]
        tensor = self._load_record(path)
        
        # Pad or truncate if needed
        tensor = self._pad_or_truncate(tensor)
        
        # Add channel dimension: [L] -> [1, L] for 1D convolutions
        tensor = tensor.unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            tensor = self.transform(tensor)
        
        if self.labels is not None:
            return tensor, self.labels[idx]
        
        return tensor


def create_ecg_dataloader(
    data_paths: List[Union[str, Path]],
    lead_name: str = "I",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    labels: Optional[List[int]] = None,
    target_length: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader for ECG data.
    
    Args:
        data_paths: List of file paths to ECG records
        lead_name: ECG lead to extract
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        labels: Optional labels for supervised learning
        target_length: Fixed sequence length (pad/truncate)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        PyTorch DataLoader instance
    """
    dataset = ECGDataset(
        data_paths=data_paths,
        lead_name=lead_name,
        labels=labels,
        target_length=target_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[4]
    
    # Test with MAT files
    mat_paths = [
        str(project_root / "data" / "for_test" / "mat" / "JS00001"),
    ]
    
    # Create dataset
    dataset = ECGDataset(
        data_paths=mat_paths,
        lead_name="I",
        target_length=5000,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")  # Expected: [1, 5000]
    
    # Create dataloader
    dataloader = create_ecg_dataloader(
        data_paths=mat_paths,
        lead_name="I",
        batch_size=2,
        target_length=5000,
    )
    
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")  # Expected: [2, 1, 5000]
        break
