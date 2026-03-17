from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.loader.ecg_dataset import ECGDataset
from module.model.feature_extractor.feature_extractor import FeatureExtractor
from module.model.base_model.transformer import HerculesTransformer
from module.model.contrastive_learning.hercules_momentum_contrast import HerculesMomentumContrast


DATA_ROOT    = Path(__file__).resolve().parent / "data" / "for_test" / "minimal_dataset" / "WFDBRecords"
LEAD_NAME    = "I"
TARGET_LEN   = 5000        # samples per record after pad/truncate
BATCH_SIZE   = 16
QUEUE_SIZE   = 256         # must be divisible by BATCH_SIZE; keep small for dev run
EMBED_DIM    = 512
D_MODEL      = 512         # must match FeatureExtractor output channels
NUM_LAYERS   = 4
NUM_HEADS    = 8
MOMENTUM     = 0.99
TEMPERATURE  = 0.07
LR           = 3e-4
EPOCHS       = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

class GaussianNoise:
    """Return two independently noised views of the same ECG signal."""
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [1, L]
        Returns:
            (view_q, view_k): each [1, L], with independent Gaussian noise
        """
        x = x.float()
        noise_q = torch.randn_like(x) * self.std
        noise_k = torch.randn_like(x) * self.std
        return x + noise_q, x + noise_k



class ContrastiveECGDataset(ECGDataset):
    def __init__(self, data_paths, lead_name, target_length, noise_std=0.05):
        super().__init__(data_paths=data_paths, lead_name=lead_name, target_length=target_length)
        self.augment = GaussianNoise(std=noise_std)

    def __getitem__(self, idx):
        tensor = super().__getitem__(idx)   # [1, L]
        view_q, view_k = self.augment(tensor)
        return view_q, view_k



def collect_record_paths(data_root: Path):
    paths = []
    for hea_file in sorted(data_root.rglob("*.hea")):
        paths.append(str(hea_file.with_suffix("")))
    return paths


def build_encoder(d_model: int, num_layers: int, num_heads: int, embed_dim: int) -> nn.Sequential:
    feature_extractor = FeatureExtractor()
    transformer = HerculesTransformer(
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        contrastive_embedding_dim=embed_dim,
    )
    return nn.Sequential(feature_extractor, transformer)



def train_one_epoch(model: HerculesMomentumContrast, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.CrossEntropyLoss,
                    device: str, epoch: int) -> float:
    model.train()
    total_loss = 0.0

    for step, (xq, xk) in enumerate(loader):
        xq = xq.float().to(device)   # [B, 1, L]
        xk = xk.float().to(device)

        logits, labels = model(xq, xk)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(f"  Epoch [{epoch}] step [{step + 1}/{len(loader)}]  loss: {loss.item():.4f}")

    return total_loss / len(loader)


def main():
    print(f"Device: {DEVICE}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    record_paths = collect_record_paths(DATA_ROOT)
    print(f"Found {len(record_paths)} records.")

    dataset = ContrastiveECGDataset(
        data_paths=record_paths,
        lead_name=LEAD_NAME,
        target_length=TARGET_LEN,
        noise_std=0.05,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,   # required: queue_size must be divisible by batch_size
    )

    encoder = build_encoder(D_MODEL, NUM_LAYERS, NUM_HEADS, EMBED_DIM)
    model = HerculesMomentumContrast(
        base_encoder=encoder,
        queue_size=QUEUE_SIZE,
        embedding_dim=EMBED_DIM,
        momentum_coefficient=MOMENTUM,
        softmax_temperature=TEMPERATURE,
    ).to(DEVICE)

    # ── Optimizer & Loss ──
    optimizer = torch.optim.AdamW(model.encoder_query.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, DEVICE, epoch)
        print(f"Epoch [{epoch}/{EPOCHS}]  avg_loss: {avg_loss:.4f}")

        checkpoint_path = CHECKPOINT_DIR / f"hercules_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "encoder_query_state": model.encoder_query.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"  Checkpoint saved → {checkpoint_path}")

    print("Pre-training complete.")


if __name__ == "__main__":
    main()
