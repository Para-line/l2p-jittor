import jittor as jt

jt.flags.use_cuda = 0

x = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
print(x)

x = jt.index_fill_(x, 0, jt.array([0, 2]), -1)
print(x)

x = jt.index_fill_(x, 1, jt.array([0, 2]), -1)
print(x)

jt.sync_all(True)
print(x)