from MatrixUtil  import Matrix,Sum,DotPro,MRelu,MatMul,ToDot,Forward,Backward

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


#def Train():
#    vec=Vector().BuildFromNative(10,[i for i in range(10)])
#    SumGraph=Graph(vec)
#    for i in range(100):
#        SumGraph.SetInputValue(xxxx)
#        SumGraph()
#       SumGraph.Backward()
