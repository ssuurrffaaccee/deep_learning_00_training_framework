from Module import *
from Optim import SimpleGradOptim as SGO
class Model:
    def __init__(self):
        self.Result=None
    def __call__(self,Input):
        GTRAIN.SetTrain()
        self.Result=self.Build(Input)
        return self.Result
    def Build(self,Input):
        return []
    def Eval(self,Input):
        #turn off DropOut
        GTRAIN.SetEval()
        self.Result=self.Build(Input)
        Forward()
        #no Backward，so need Clear manually
        GDataNode.ClearConsts()
        GOperator.Clear()
        return self.Result
class SumFFN(Model):
    def __init__(self,Layers):
        for l in Layers:
            assert l!=0
        self.Layers=Layers
        self.FNN=FNNRelu(Layers)
    def Build(self,Input):
        assert self.FNN.Layers[0]==Input.H
        assert Input.W==1
        return Sum(self.FNN(Input))

