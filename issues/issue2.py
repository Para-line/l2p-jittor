import jittor as jt

jt.flags.use_cuda = 1

x = jt.init.uniform((10, 768), low = -1, high= 1)

print(x.sum())
print(jt.sum(x))

print(x.sum().item())
print(jt.sum(x).item())

loop_sum = 0.0
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        loop_sum += x[i, j]

print(f"Sum calculated with loop: {loop_sum}")

jt.sync_all(True)

print("=================================")

print(x.sum())
print(jt.sum(x))

print(x.sum().item())
print(jt.sum(x).item())

loop_sum = 0.0
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        loop_sum += x[i, j]

print(f"Sum calculated with loop: {loop_sum}")