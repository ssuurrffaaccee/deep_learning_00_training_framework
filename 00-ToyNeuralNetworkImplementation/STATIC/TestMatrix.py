from  MatrixUtil import * 
def TestSum1():
    Vec=Vector().BuildFromNative(10,[i for i in range(10)])
    Result=Sum(DotPro(Vec,Vec))
    Forward()
    Backward()
    ToDot("dot/MatrixT1.gv")
def TestMatMul():
    Vec1=Vector().BuildFromNative(10,[i for i in range(10)])
    Vec2=Matrix().BuildFromNative(1,10,[i for i in range(10)])
    Result=MatMul(Vec2,Vec1)
    Forward()
    Backward()
    ToDot("dot/MatrixT2.gv")

def TestMatMul2():
    Vec1=Vector().BuildFromNative(2,[i+1 for i in range(2)])
    Vec2=Matrix().BuildFromNative(3,2,[i+1 for i in range(3*2)])
    Result=Sum(MatMul(Vec2,Vec1))
    Forward()
    Backward()
    ToDot("dot/MatrixT3.gv")
def TestRelu():
    Vec1=Vector().BuildFromNative(2,[i-5 for i in range(2)])
    Vec2=Matrix().BuildFromNative(3,2,[i for i in range(3*2)])
    Result=Sum(MRelu(MatMul(Vec2,Vec1)))
    Forward()
    Backward()
    ToDot("dot/MatrixT4.gv")
#TestSum1()
#TestMatMul()
#TestMatMul2()
TestRelu()