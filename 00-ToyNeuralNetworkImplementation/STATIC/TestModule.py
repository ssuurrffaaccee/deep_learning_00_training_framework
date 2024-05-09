from Module import *
def Test():
    Vec=Vector().BuildFromNative(3,[i for i in range(3)])
    Result=Sum(FNNRelu([3,3,3,3,3])(Vec))
    Forward()
    Backward()
    ToDot("dot/FNN.gv")
Test()