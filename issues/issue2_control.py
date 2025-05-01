import torch
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

low = -1
high = 1
x = (high - low) * torch.rand(10, 768, device=device) + low

print("\n--- Before sync ---")
print("x.sum() tensor:", x.sum())
print("torch.sum(x) tensor:", torch.sum(x))

sum_val_1 = x.sum().item()
sum_val_2 = torch.sum(x).item()
print("x.sum().item():", sum_val_1)
print("torch.sum(x).item():", sum_val_2)


print("\nCalculating sum using a Python loop (before sync):")

x_numpy = x.cpu().numpy()

loop_sum_before = 0.0
for i in range(x_numpy.shape[0]):
    for j in range(x_numpy.shape[1]):
        loop_sum_before += x_numpy[i, j]
print(f"Sum calculated with loop (from CPU copy): {loop_sum_before}")
print(f"Is PyTorch sum close to loop sum? {np.isclose(sum_val_2, loop_sum_before)}")


if device.type == 'cuda':
    torch.cuda.synchronize()
    print("\ntorch.cuda.synchronize() called.\n")
else:
    print("\nNo explicit sync needed for CPU.\n")


print("--- After sync ---")
print("x.sum() tensor:", x.sum())
print("torch.sum(x) tensor:", torch.sum(x))


sum_val_3 = x.sum().item()
sum_val_4 = torch.sum(x).item()
print("x.sum().item():", sum_val_3)
print("torch.sum(x).item():", sum_val_4)


print("\nCalculating sum using a Python loop (after sync):")
x_numpy_after = x.cpu().numpy()
loop_sum_after = 0.0
for i in range(x_numpy_after.shape[0]):
    for j in range(x_numpy_after.shape[1]):
        loop_sum_after += x_numpy_after[i, j]
print(f"Sum calculated with loop (from CPU copy): {loop_sum_after}")
print(f"Is PyTorch sum close to loop sum? {np.isclose(sum_val_4, loop_sum_after)}")

print("\n=================================")