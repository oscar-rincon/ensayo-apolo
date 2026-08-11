import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ======================================================
# Configuration
# ======================================================

USE_GPU = False          # Set False to force CPU
SEED = 42
EPOCHS = 70_001
LR = 1e-3

torch.manual_seed(SEED)

# ======================================================
# Results folder
# ======================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results_mlp", timestamp)
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print("Simple Neural Network Training")
print("=" * 60)
print(f"Results directory : {results_dir}")

# ======================================================
# Device
# ======================================================

if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device            : {device}")

if device.type == "cuda":
    print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version      : {torch.version.cuda}")

# ======================================================
# Generate data
# ======================================================

x = torch.linspace(
    -2 * torch.pi,
    2 * torch.pi,
    500,
).view(-1, 1)

y = torch.sin(15 * x)

x = x.to(device)
y = y.to(device)

# ======================================================
# Neural Network
# ======================================================

model = nn.Sequential(
    nn.Linear(1, 125),
    nn.Tanh(),
    nn.Linear(125, 125),
    nn.Tanh(),
    nn.Linear(125, 125),
    nn.Tanh(),    
    nn.Linear(125, 125),
    nn.Tanh(),   
    nn.Linear(125, 125),
    nn.Tanh(),           
    nn.Linear(125, 125),
    nn.Tanh(),        
    nn.Linear(125, 1),
).to(device)

# ======================================================
# Loss and optimizer
# ======================================================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================================================
# Model information
# ======================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"Total parameters  : {total_params}")
print(f"Trainable params  : {trainable_params}")

# ======================================================
# Training
# ======================================================

loss_history = []

print("\nStarting training...\n")

start = time.perf_counter()

for epoch in range(EPOCHS):

    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d}/{EPOCHS-1} | "
            f"Loss = {loss.item():.8e}"
        )

# Wait for GPU to finish before stopping timer
if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("\nTraining completed.")
print(f"Training time : {elapsed:.4f} s")
print(f"Final loss    : {loss_history[-1]:.8e}")

# ======================================================
# Prediction
# ======================================================

model.eval()

with torch.no_grad():
    y_pred = model(x)

# ======================================================
# Move tensors to CPU
# ======================================================

x_cpu = x.cpu()
y_cpu = y.cpu()
y_pred_cpu = y_pred.cpu()

# ======================================================
# Save model
# ======================================================

torch.save(
    model.state_dict(),
    os.path.join(results_dir, "model.pt"),
)

with open(os.path.join(results_dir, "loss_history.txt"), "w") as f:
    for value in loss_history:
        f.write(f"{value:.12e}\n")

# ======================================================
# Prediction plot
# ======================================================

plt.figure(figsize=(8, 4))

plt.plot(
    x_cpu.numpy(),
    y_cpu.numpy(),
    label="True function",
)

plt.plot(
    x_cpu.numpy(),
    y_pred_cpu.numpy(),
    "--",
    linewidth=2,
    label="Neural Network",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Approximation")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(results_dir, "prediction.pdf"),
    dpi=300,
)

plt.close()

# ======================================================
# Loss plot
# ======================================================

plt.figure(figsize=(8, 4))

plt.plot(loss_history)

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(results_dir, "loss.pdf"),
    dpi=300,
)

plt.close()

# ======================================================
# Training summary
# ======================================================

summary_path = os.path.join(
    results_dir,
    "training_summary.txt",
)

with open(summary_path, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("NEURAL NETWORK TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("GENERAL INFORMATION\n")
    f.write("-" * 70 + "\n")

    f.write(f"Date                : {datetime.now()}\n")
    f.write(f"Results directory   : {results_dir}\n")
    f.write(f"Device              : {device}\n")

    if device.type == "cuda":
        f.write(
            f"GPU                 : "
            f"{torch.cuda.get_device_name(0)}\n"
        )
        f.write(
            f"CUDA version        : "
            f"{torch.version.cuda}\n"
        )

    f.write("\n")

    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 70 + "\n")

    f.write(f"Seed                : {SEED}\n")
    f.write(f"Epochs              : {EPOCHS}\n")
    f.write(f"Learning rate       : {LR}\n")
    f.write(f"Optimizer           : Adam\n")
    f.write(f"Loss function       : MSELoss\n")

    f.write("\n")

    f.write("DATASET\n")
    f.write("-" * 70 + "\n")

    f.write(f"Samples             : {len(x_cpu)}\n")
    f.write(
        f"x range             : "
        f"[{x_cpu.min().item():.3f}, "
        f"{x_cpu.max().item():.3f}]\n"
    )
    f.write("Target function     : sin(15x)\n")

    f.write("\n")

    f.write("MODEL\n")
    f.write("-" * 70 + "\n")

    f.write(str(model))
    f.write("\n\n")

    f.write(f"Total parameters    : {total_params}\n")

    f.write("\n")

    f.write("RESULTS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Training time (s)   : "
        f"{elapsed:.6f}\n"
    )

    f.write(
        f"Final loss          : "
        f"{loss_history[-1]:.12e}\n"
    )

    f.write(
        f"Minimum loss        : "
        f"{min(loss_history):.12e}\n"
    )

    f.write(
        f"Maximum loss        : "
        f"{max(loss_history):.12e}\n"
    )

# ======================================================
# Console summary
# ======================================================

print("\n" + "=" * 60)
print("Training finished successfully")
print("=" * 60)
print(f"Training time : {elapsed:.4f} s")
print(f"Final loss    : {loss_history[-1]:.6e}")

print("\nGenerated files:")

files = [
    "training_summary.txt",
    "prediction.pdf",
    "loss.pdf",
    "loss_history.txt",
    "model.pt",
]
 
print(f"\nAll results saved in:\n{results_dir}")