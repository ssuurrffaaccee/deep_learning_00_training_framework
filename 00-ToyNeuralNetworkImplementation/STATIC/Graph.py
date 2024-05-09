from MatrixUtil  import Matrix,Sum,DotPro,MRelu,MatMul,ToDot,Forward,Backward
#all need delete
class Graph:
    def __init__(self):
        self.Input=None
        self.Output=None
    def __call__(self,Input):
        self.Input=Input
        self.Output=self.Build(Input)
    def Build(self,Input):
        return Sum(Input)
    def Backward(self):
        Backward()
    def Forward(self):
        Forward()
    def SetInputValue(self,Native):
        self.Input.SetValue(Native)

def FNN(Input,Layers):
    for N in Layers:
        assert N!=0
    assert Input.H==Layers[0]
    Result=Input
    for Index in range(len(Layers)-1):
        H=Layers[Index+1]
        W=Layers[Index]
        Result=MRelu(MatMul(Matrix().BuildFromNative(H,W,[i+1 for i in range(H*W)]),Result))
    return Result
#def Train():
#    vec=Vector().BuildFromNative(10,[i for i in range(10)])
#    SumGraph=Graph(vec)
#    for i in range(100):
#        SumGraph.SetInputValue(xxxx)
#        SumGraph()
#       SumGraph.Backward()
