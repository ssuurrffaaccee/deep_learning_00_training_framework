from Function import *
from Tensor import InputTensor
from Module import Forward
import numpy as np
def TestNorm2():
    a=InputTensor(np.array([1,2]))
    b=InputTensor(np.array([1,5]))
    x=Norm2Dis(a,b)
    Forward()
    print(x.Data)