import torch
import torch.nn as nn
import ncdl
import ncdl.nn as ncnn
from ncdl.modules.autoencoder import DoubleConv, OutConv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SmallQuincunxAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp = ncdl.Lattice('cp')
        S = np.array([[1, 1], [1, -1]], dtype='int')
        self.conv1 = DoubleConv(self.cp, 1, 16)
        self.down1 = ncnn.LatticeDownsample(self.cp, S)
        qc = self.down1.down_lattice
        self.conv2 = DoubleConv(qc, 16, 32)
        self.down2 = ncnn.LatticeDownsample(qc, S)
        cp2 = self.down2.down_lattice
        self.conv3 = DoubleConv(cp2, 32, 32)
        self.up1 = ncnn.LatticeUpsample(cp2, S)
        qc2 = self.up1.up_lattice
        self.conv4 = DoubleConv(qc2, 32, 16)
        self.up2 = ncnn.LatticeUpsample(qc2, S)
        cp3 = self.up2.up_lattice
        self.conv5 = DoubleConv(cp3, 16, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        lt = self.cp(x)
        lt = self.conv1(lt); lt_ref1 = lt
        lt = self.down1(lt)
        lt = self.conv2(lt); lt_ref2 = lt
        lt = self.down2(lt)
        lt = self.conv3(lt)
        lt = self.up1(lt)
        lt = ncdl.pad_like(lt, lt_ref2)
        lt = self.conv4(lt)
        lt = self.up2(lt)
        lt = ncdl.pad_like(lt, lt_ref1)
        lt = self.conv5(lt)
        return torch.sigmoid(self.out(lt))

transform = transforms.ToTensor()
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = 6000
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def evaluate(model, loader):
    model.eval()
    total_l1, total_l2, total_ssim, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            reconstructed = model(images)
            total_l1 += nn.functional.l1_loss(reconstructed, images, reduction='sum').item()
            total_l2 += nn.functional.mse_loss(reconstructed, images, reduction='sum').item()
            total_ssim += ssim_metric(reconstructed, images).item() * images.size(0)
            n += images.size(0)
    avg_l1 = total_l1 / (n * 28 * 28)
    avg_l2 = total_l2 / (n * 28 * 28)
    psnr = 10 * np.log10(1.0 / avg_l2) if avg_l2 > 0 else float('inf')
    avg_ssim = total_ssim / n
    return {'l1': avg_l1, 'l2': avg_l2, 'psnr': psnr, 'ssim': avg_ssim}

model = SmallQuincunxAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 20
history = []
best_val_l1 = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    total_train_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = nn.functional.l1_loss(reconstructed, images)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * images.size(0)
    train_l1 = total_train_loss / len(train_set)

    val_metrics = evaluate(model, val_loader)
    scheduler.step(val_metrics['l1'])
    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
          f"train L1: {train_l1:.4f} | val L1: {val_metrics['l1']:.4f} | "
          f"val L2: {val_metrics['l2']:.6f} | val PSNR: {val_metrics['psnr']:.2f} | "
          f"val SSIM: {val_metrics['ssim']:.4f} | lr: {current_lr:.2e}")

    history.append({
        'epoch': epoch + 1,
        'train_l1': train_l1,
        'val_l1': val_metrics['l1'],
        'val_l2': val_metrics['l2'],
        'val_psnr': val_metrics['psnr'],
        'val_ssim': val_metrics['ssim'],
        'learning_rate': current_lr,
        'epoch_time_seconds': epoch_time,
    })

    if val_metrics['l1'] < best_val_l1:
        best_val_l1 = val_metrics['l1']
        torch.save(model.state_dict(), 'model_best.pt')
        print(f"  -> new best model saved (val L1: {best_val_l1:.4f})")

model.load_state_dict(torch.load('model_best.pt'))
test_metrics = evaluate(model, test_loader)
print(f"\nFinal test set results (best model):")
print(f"  L1:   {test_metrics['l1']:.4f}")
print(f"  L2:   {test_metrics['l2']:.6f}")
print(f"  PSNR: {test_metrics['psnr']:.2f}")
print(f"  SSIM: {test_metrics['ssim']:.4f}")

results = {
    'training_history': history,
    'test_metrics': test_metrics,
    'num_epochs': num_epochs,
    'best_val_l1': best_val_l1,
}
with open('training_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

torch.save(model.state_dict(), 'model_final.pt')
print("\nSaved: model_best.pt, model_final.pt, training_metrics.json")