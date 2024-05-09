from Model import FNNModel,MinstCNNModel,SimpleFakeRNNModel,SimpleFakeMultiRNNModel,SimpleGNNModel
from Tensor import InputTensor,ParameterTensor
import numpy as np
def TestFNNModel():
    FNN=FNNModel([2,4,5,1])
    In=InputTensor(np.array([1,2]))
    Out=FNN(In)
    FNN.OneRound()
    print(FNN.GetOuputData())
    In=InputTensor(np.array([2,2]))
    Out=FNN(In)
    FNN.OneRound()
    print(FNN.GetOuputData())
def TestCNN():
    MD=MinstCNNModel()
    IN=InputTensor(np.random.randn(28,28)*250)
    Output=MD(IN)
    MD.OneRound()
    print(Output.Data)
def TestRNN():
    SFR=SimpleFakeRNNModel(10,20)
    IN1=InputTensor(np.random.randn(20,1))
    InitHidden=InputTensor(np.zeros([10,1]))
    Hidden,WordPro=SFR(IN1,InitHidden)
    SFR.OneRound()
    print(Hidden.Data)
    print(np.argmax(WordPro.Data))
def TestRNNGenenrator(Length):
    SFR=SimpleFakeRNNModel(10,20)
    IN1=InputTensor(np.random.randn(20,1))
    Hidden=InputTensor(np.zeros([10,1]))
    WordPros=[]
    Hiddens=[]
    for i in range(Length):
        IN=InputTensor(np.random.randn(20,1))
        Hidden,WordPro=SFR(IN,Hidden)
        WordPros.append(WordPro)
        Hiddens.append(Hidden)
    SFR.Forward()
    for Pro in WordPros:
        #print(Pro.Data)
        print(np.argmax(Pro.Data))
    #for Hid in Hiddens:
    #    print(Hidden.Data)
    #print(Hidden.Data)
    #print(np.argmax(WordPro.Data))
def TestMultiRNNGenenrator(Length):
    LayerNum=30
    SFR=SimpleFakeMultiRNNModel(10,20,LayerNum)
    IN1=InputTensor(np.random.randn(20,1))
    Hiddens=[InputTensor(np.zeros([10,1])) for i in range(LayerNum)]
    WordPros=[]
    #Hiddens=[]
    for i in range(Length):
        IN=InputTensor(np.random.randn(20,1))
        Hidden,WordPro=SFR(IN,Hiddens)
        WordPros.append(WordPro)
        #Hiddens.append(Hidden)
    SFR.Forward()
    for Pro in WordPros:
        #print(Pro.Data)
        print(np.argmax(Pro.Data))
    #for Hid in Hiddens:
    #    print(Hidden.Data)
    #print(Hidden.Data)
    #print(np.argmax(WordPro.Data))
def TestGNN():
    NodeEmbeddingLen=2
    Node1=InputTensor(np.random.randn(4,NodeEmbeddingLen))
    print(Node1.Data)
    Adjacent=InputTensor(np.random.binomial(1,0.5,[4,4]))
    print(Adjacent.Data)
    G1=SimpleGNNModel(NodeEmbeddingLen)
    G2=SimpleGNNModel(NodeEmbeddingLen)
    Node2=G1(Node1,Adjacent)
    Node3=G2(Node2,Adjacent)
    G1.Forward()
    print(Node2.Data)
    print(Node3.Data)
#TestFNNModel()
#TestCNN()
#TestRNN()
#TestRNNGenenrator(10000)
#TestMultiRNNGenenrator(100)
TestGNN()