import tvm
import tvm.testing
from tvm import te
import numpy
import timeit


M = 4096
K = 1024
N = 128

dtype = "float32"

target = "llvm -mcpu=core-avx2"
dev = tvm.device(target, 0)#создает объект, который представляет первое устройство указанного типа и может использоваться для выполнения операций в TVM.

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
a_transposed = tvm.nd.array(numpy.random.rand(K, M).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)


#answer = numpy.dot(a_transposed.numpy(), b.numpy())#произведение массивов

# Algorithm
k = te.reduce_axis((0, K), "k")#используется для указания осей (или размерностей), по которым будет выполняться операция редукции? уменьшение размерности
A = te.placeholder((M, K), name="A")
#A_transposed = te.placeholder((K, M), name="A_transposed")
#A_transposed = te.compute((K, M), lambda k, m: A[m, k], name='A_transposed')
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# bn = 64
# s = te.create_schedule(C.op)

# mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# s[C].vectorize(ni)

# s[C].reorder(mo, no, k, mi, ni)

#__________________________
# bn = 64
# s = te.create_schedule(C.op)

# mo = C.op.axis[0]
# no, ni = s[C].split(C.op.axis[1], bn)

# s[C].vectorize(ni)

# s[C].reorder(mo, no, k, ni)

#________________________________________________
bn = 64
kfactor = 16
s = te.create_schedule(C.op)

mo = C.op.axis[0]
no, ni = s[C].split(C.op.axis[1], bn)

(kaxis,) = s[C].op.reduce_axis 
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].vectorize(ni)

s[C].reorder(mo, no, ko, ki, ni)

#______________________________________-
# no = C.op.axis[1]
# mo, mi = s[C].split(C.op.axis[0], bn)

# s[C].vectorize(mi)

# s[C].reorder(mo, no, k, mi)


func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func
func.export_library("gemm_copy" + ".so")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
#tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt: %f" % evaluator(a, b, c).mean)

