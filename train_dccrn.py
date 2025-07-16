import os
import torch
import logging
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from dccrn_model import DCCRN
from torch import optim
import torchaudio

# 🛑 Suppress Warnings
warnings.filterwarnings("ignore")

# 🪵 Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📦 Dataset class
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch in dataset size"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(self.noisy_files[idx])
        clean, _ = torchaudio.load(self.clean_files[idx])
        return noisy, clean

# ⚙️ Configuration class
class Args:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.clean_dir = os.path.join(base_path, "dataset", "clean")
        self.noisy_dir = os.path.join(base_path, "dataset", "noisy")
        self.rnn_layers = 2
        self.rnn_units = 256
        self.masking_mode = 'E'
        self.use_clstm = True
        self.use_cbn = False
        self.kernel_num = [32, 64, 128, 256, 256, 256]
        self.batch_size = 1
        self.epochs = 300
        self.lr = 1e-3
        self.weight_decay = 1e-7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_mode = 'SI-SNR'  # or 'MSE'

# 🔍 Loss computation wrapper
def compute_loss(model, enhanced, target, mode='SI-SNR'):
    # Add channel dimension if missing
    if enhanced.dim() == 2:
        enhanced = enhanced.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    return model.loss(enhanced, target, loss_mode=mode)

# 🏋️ Train one epoch
def train_epoch(model, dataloader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    
    for i, (noisy, clean) in enumerate(pbar):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        _, enhanced = model(noisy)

        # Match length
        min_len = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :min_len]
        clean = clean[..., :min_len]

        loss = compute_loss(model, enhanced, clean, args.loss_mode)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{total_loss / (i + 1):.4f}'})

    return total_loss / len(dataloader)

# 🚀 Main training loop
def main():
    args = Args()
    logger.info("🚀 DCCRN Training Config")
    logger.info(f"Clean Dir  : {args.clean_dir}")
    logger.info(f"Noisy Dir  : {args.noisy_dir}")
    logger.info(f"Device     : {args.device}")
    logger.info(f"Loss Mode  : {args.loss_mode}")
    logger.info("=" * 40)

    model = DCCRN(
        rnn_layers=args.rnn_layers,
        rnn_units=args.rnn_units,
        masking_mode=args.masking_mode,
        use_clstm=args.use_clstm,
        use_cbn=args.use_cbn,
        kernel_num=args.kernel_num
    ).to(args.device)

    # Dataset & Dataloader
    dataset = AudioDataset(args.noisy_dir, args.clean_dir)
    logger.info(f"✅ Found {len(dataset)} audio pairs")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 🔁 Training
    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch + 1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, args.device, epoch, args)

        logger.info(f"✅ Avg Loss: {avg_loss:.4f}")

    # 💾 Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "dccrn_final.pth"))
    logger.info("🎉 Training complete. Model saved at checkpoints/dccrn_final.pth")

if __name__ == "__main__":
    main()
