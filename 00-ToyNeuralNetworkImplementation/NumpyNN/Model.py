from Module import *
class Model:
    def __init__(self,Name=""):
        self.Name=Name
        self.Output=None
    def __call__(self,Input,Train=True):
        if Train:
            self.SetTrain()
        else:
            self.SetEval()
        self.Build(Input)
        return self.Output
    def Build(self,Input):
        pass
    def Forward(self):
        Forward()
    def Backward(self,Clear=True):
        Backward()
        if Clear:
            self.Clear()
    def GetOuputData(self):
        assert type(self.Output)!=type(None)
        return self.Output.Data
    def OneRound(self):
        self.Forward()
        self.Backward()
    def SetEval(self):
        TRAIN.Eval()
    def SetTrain(self):
        TRAIN.Train()
    def Clear(self):
        ClearDataNode()
        ClearOperator()
class FNNModel(Model):
    def __init__(self,Layers):
        self.Layers=Layers
        self.FFN=FNN(Layers,type(self).__name__,Dropout=0.0)
    def Build(self,Input):
        self.Output=self.FFN(Input)
class MinstCNNModel(Model):
    def __init__(self):
        self.CNN1=Simple2DCNN([3,3])
        self.CNN2=Simple2DCNN([4,4])
        self.FNN1=FNN([5*5,10,10])
    def Build(self,Input):
        CNN1Output=MaxPool(self.CNN1(Input,[28,28]),[2,2])
        CNN2Output=MaxPool(self.CNN2(CNN1Output,[13,13]),[2,2])
        Output=self.FNN1(Flatten(CNN2Output))
        self.Output=Softmax(Output)
class FNNMinstModel(Model):
    def __init__(self,FirstLayer,Layers,LastLayer):
        self.Layers=Layers
        self.FNN=ResidulFNN(FirstLayer,Layers,LastLayer,Activation=Sigmoid,Dropout=0.1)
    def Build(self,Input):
        if TRAIN():
            self.Output=self.FNN(Input)
        else:
            self.Output=SoftmaxForEval(self.FNN(Input))
class SimpleFakeRNNModel(Model):
    def __init__(self,HiddenSize,InputSize):
        self.InputSize=InputSize
        self.HiddenSize=HiddenSize
        self.RNN=SimpleRNN(HiddenSize,InputSize)
        self.FakeEmbedding=GetParameter(100,HiddenSize)
    def GetOuputData(self):
        assert type(self.Output)!=type(None)
        return self.Output[1].Data
    def __call__(self,Input,Hidden,Train=True):
        if Train:
            self.SetTrain()
        else:
            self.SetEval()
        self.Build(Input,Hidden)
        return self.Output[0],self.Output[1]
    def Build(self,Input,Hidden):
        NewHidden=self.RNN(Input,Hidden)
        Output=Softmax(MatMul(self.FakeEmbedding,NewHidden))
        self.Output=[NewHidden,Output]
class SimpleFakeMultiRNNModel(Model):
    def __init__(self,HiddenSize,InputSize,LayerNum):
        self.InputSize=InputSize
        self.HiddenSize=HiddenSize
        self.LayerNum=LayerNum
        self.RNN=SimpleMultiLayerRNN(HiddenSize,InputSize,LayerNum)
        self.FakeEmbedding=GetParameter(100,HiddenSize)
    def GetOuputData(self):
        assert type(self.Output)!=type(None)
        return self.Output[1].Data
    def __call__(self,Input,Hidden,Train=True):
        if Train:
            self.SetTrain()
        else:
            self.SetEval()
        self.Build(Input,Hidden)
        return self.Output[0],self.Output[1]
    def Build(self,Input,Hiddens):
        NewHiddens=self.RNN(Input,Hiddens)
        Output=Softmax(MatMul(self.FakeEmbedding,NewHiddens[-1]))
        self.Output=[NewHiddens,Output]
class SimpleGNNModel(Model):
    def __init__(self,NodeEmbeddingLength):
        self.NodeEmbeddingLength=NodeEmbeddingLength
        self.GNN=SimpleGNN(NodeEmbeddingLength)
    def __call__(self,Input,Adjacent,Train=True):
        if Train:
            self.SetTrain()
        else:
            self.SetEval()
        self.Build(Input,Adjacent)
        return self.Output
    def Build(self,Node,Adjacent):
        self.Output=self.GNN(Node,Adjacent)