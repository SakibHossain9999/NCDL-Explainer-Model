import json
import torch
import torch.nn as nn
import ncdl
import ncdl.nn as ncnn
from ncdl.modules.autoencoder import DoubleConv, OutConv
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

with open('training_metrics.json', 'r') as f:
    results = json.load(f)

history = results['training_history']
test_metrics = results['test_metrics']

epochs = [row['epoch'] for row in history]
train_l1 = [row['train_l1'] for row in history]
val_l1 = [row['val_l1'] for row in history]
val_l2 = [row['val_l2'] for row in history]
val_ssim = [row['val_ssim'] for row in history]
val_psnr = [row['val_psnr'] for row in history]

# ============================================================
# File 1: quantitative graphs + final test results, one image
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

axes[0].plot(epochs, train_l1, label='Train L1', marker='o')
axes[0].plot(epochs, val_l1, label='Validation L1', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('L1 loss')
axes[0].set_title('L1 loss over training')
axes[0].legend()

axes[1].plot(epochs, val_l2, color='red', marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('L2 loss')
axes[1].set_title('Validation L2 over training')

axes[2].plot(epochs, val_ssim, color='green', marker='o')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('SSIM')
axes[2].set_title('Validation SSIM over training')

axes[3].plot(epochs, val_psnr, color='purple', marker='o')
axes[3].set_xlabel('Epoch')
axes[3].set_ylabel('PSNR (dB)')
axes[3].set_title('Validation PSNR over training')

test_text = (f"Final test set results (best model): "
             f"L1 = {test_metrics['l1']:.4f}   "
             f"L2 = {test_metrics['l2']:.6f}   "
             f"PSNR = {test_metrics['psnr']:.2f}   "
             f"SSIM = {test_metrics['ssim']:.4f}")
fig.suptitle(test_text, fontsize=11, y=1.02)

plt.tight_layout()
plt.savefig('quantitative_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: quantitative_results.png")

# ============================================================
# File 2: qualitative results, real input/reconstruction pairs
# ============================================================
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallQuincunxAutoencoder().to(device)
model.load_state_dict(torch.load('model_best.pt'))
model.eval()

transform = transforms.ToTensor()
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_examples = 8
fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 1.5, 3.5))

with torch.no_grad():
    for i in range(num_examples):
        image, label = test_set[i]
        image = image.unsqueeze(0).to(device)
        reconstructed = model(image)

        axes[0, i].imshow(image.cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f"input: {label}", fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed.cpu().squeeze(), cmap='gray')
        axes[1, i].set_title("reconstructed", fontsize=9)
        axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('qualitative_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: qualitative_results.png")