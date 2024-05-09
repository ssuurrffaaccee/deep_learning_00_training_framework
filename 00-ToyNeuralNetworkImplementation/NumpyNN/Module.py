from Function import Add,MatMul,Minus,Mul,Select,Concat,Relu,Sum,Norm2Dis,Sigmoid,Dropout,TRAIN,SoftmaxEntropy,SoftmaxForEval,MaxPool,Flatten,Entropy,Softmax,Tanh
from Tensor import GetParameter
from OperatorBase import Forward,Backward,ClearDataNode,ClearOperator
import numpy as np
class FNN:
    def __init__(self,Layers,Name="",Activation=Relu,Dropout=0.0):
        self.M=[]
        self.b=[]
        self.ActivFn=Activation
        self.Layers=Layers
        self.Dropout=Dropout
        self.Name="FNN @ "+Name if Name=="" else "FNN"
        assert len(Layers)>1
        for l in Layers:
            assert l!=0
        self.Init()
    def Init(self):
        MSizes=[]
        for Ind in range(len(self.Layers)-1):
            MSizes.append((self.Layers[Ind+1],self.Layers[Ind]))
        for Ms in MSizes:
            self.M.append(GetParameter(Ms[0],Ms[1]))
            self.b.append(GetParameter(Ms[0],1))
    def __call__(self,Input):
        Result=Input
        for Indx in range(len(self.M)):
            if self.Dropout==0.0:
                Result=self.ActivFn(
                                    Add(
                                        MatMul(
                                                self.M[Indx],
                                                Result,
                                                self.Name
                                               ),
                                        self.b[Indx],
                                        self.Name),
                                    self.Name
                                    )
            else:
                Result=Dropout(self.ActivFn(
                                           Add(
                                               MatMul(
                                                       self.M[Indx],
                                                       Result,
                                                       self.Name
                                                      ),
                                               self.b[Indx],
                                               self.Name),
                                           self.Name
                                           ),
                                self.Dropout
                                )               
        return Result
class ResidulFNN:
    def __init__(self,FirstLyaer,Layers,LastLayer,Name="",Activation=Relu,Dropout=0.0):
        self.M=[]
        self.b=[]
        self.FirstLyaer=FirstLyaer
        self.LastLayer=LastLayer
        self.ActivFn=Activation
        self.Layers=Layers
        self.Dropout=Dropout
        self.Name="FNN @ "+Name if Name=="" else "FNN"
        assert len(Layers)>1
        for l in Layers:
            assert l!=0
        self.Init()
    def Init(self):
        MSizes=[]
        #firse
        self.M.append(GetParameter(self.Layers[0],self.FirstLyaer))
        self.b.append(GetParameter(self.Layers[0],1))

        for Ind in range(len(self.Layers)-1):
            MSizes.append((self.Layers[Ind+1],self.Layers[Ind]))
        for Ms in MSizes:
            self.M.append(GetParameter(Ms[0],Ms[1]))
            self.b.append(GetParameter(Ms[0],1))
        self.M.append(GetParameter(self.LastLayer,self.Layers[-1]))
        self.b.append(GetParameter(self.LastLayer,1))
    def __call__(self,Input):
        First=self.ActivFn(
                    Add(
                        MatMul(
                                self.M[0],
                                Input,
                                self.Name
                               ),
                        self.b[0],
                        self.Name),
                    self.Name
                    )
        Result=First
        for Indx in range(len(self.M[1:-1])):
            if Dropout==0:
                ImediateResult=self.ActivFn(
                                    Add(
                                        MatMul(
                                                self.M[Indx+1],
                                                Result,
                                                self.Name
                                               ),
                                        self.b[Indx+1],
                                        self.Name),
                                    self.Name
                                    )
                Result=Add(ImediateResult,Result)
            else:
                ImediateResult=Dropout(self.ActivFn(
                                           Add(
                                               MatMul(
                                                       self.M[Indx+1],
                                                       Result,
                                                       self.Name
                                                      ),
                                               self.b[Indx],
                                               self.Name),
                                           self.Name
                                           ),
                                self.Dropout
                                )
                Result=Add(ImediateResult,Result)
        Last=self.ActivFn(
                    Add(
                        MatMul(
                                self.M[-1],
                                Result,
                                self.Name
                               ),
                        self.b[-1],
                        self.Name),
                    self.Name
                    )
        Result=Last               
        return Result                
class Simple2DCNN:
    def __init__(self,KernelSize,Activation=Relu):
        self.KernelSize=KernelSize
        self.Kernel=GetParameter(self.KernelSize[0],self.KernelSize[1])
        self.ActivFn=Activation
    def __call__(self,Input,InputSize):
        OutputH=InputSize[0]-self.KernelSize[0]+1
        OutputW=InputSize[1]-self.KernelSize[1]+1
        Indexs=[]
        for x in range(OutputH):
            for y in range(OutputW):
                #print((x,x+self.KH-1,y,y+self.KW-1))
                Indexs.append((x,x+self.KernelSize[0],y,y+self.KernelSize[1]))
        Tensors=[]
        for Ind in Indexs:
            Tensors.append(Sum(Mul(Select(Input,Ind),self.Kernel)))
        return self.ActivFn(Concat(Tensors,Shape=(OutputH,OutputW)))
class SimpleRNN:
    def __init__(self,HiddenSize,InputSize,Activation=Tanh):
        self.HiddenSize=HiddenSize
        self.InputSize=InputSize
        self.InputToHiddennM=GetParameter(self.HiddenSize,self.InputSize)
        self.HiddenToHiddenM=GetParameter(self.HiddenSize,self.HiddenSize)
        self.Bias=GetParameter(self.HiddenSize,1)
        self.ActivFn=Activation
    def __call__(self,Input,Hidden):
        NewMemory=MatMul(self.InputToHiddennM,Input)
        OldMemory=MatMul(self.HiddenToHiddenM,Hidden)
        NewHidden=Add(Add(NewMemory,OldMemory),self.Bias)
        return self.ActivFn(NewHidden)
class SimpleMultiLayerRNN:
    def __init__(self,HiddenSize,InputSize,LayerNum,Activation=Tanh):
        self.HiddenSize=HiddenSize
        self.InputSize=InputSize
        self.LayerNum=LayerNum
        self.InputToHiddennM=[GetParameter(self.HiddenSize,self.InputSize)]
        self.InputToHiddennM.extend([GetParameter(self.HiddenSize,self.HiddenSize) for i in range(self.LayerNum)if i!=0])
        self.HiddenToHiddenM=[GetParameter(self.HiddenSize,self.HiddenSize) for i in range(self.LayerNum)]
        self.Bias=[GetParameter(self.HiddenSize,1) for i in range(self.LayerNum)]
        self.ActivFn=Activation
    def __call__(self,Input,Hiddens):
        assert len(Hiddens)==self.LayerNum
        OuputHiddens=[]
        #First Layer
        NewMemory=MatMul(self.InputToHiddennM[0],Input)
        OldMemory=MatMul(self.HiddenToHiddenM[0],Hiddens[0])
        NewHidden=Add(Add(NewMemory,OldMemory),self.Bias[0])
        OuputHiddens.append(NewHidden)
        InterInput=NewHidden
        for i in range(self.LayerNum):
            print(i)
            if i==0:
                continue
            NewMemory=MatMul(self.InputToHiddennM[i],InterInput)
            OldMemory=MatMul(self.HiddenToHiddenM[i],Hiddens[i])
            InterInput=self.ActivFn(Add(Add(NewMemory,OldMemory),self.Bias[i]))
            OuputHiddens.append(InterInput)
        return OuputHiddens

class SimpleGNN:
    def __init__(self,NodeEmbeddingLength,Activation=Tanh):
        self.NodeEmbeddingLength=NodeEmbeddingLength
        self.M=GetParameter(self.NodeEmbeddingLength,self.NodeEmbeddingLength)
        #self.b=GetParameter(self.NodeEmbeddingLength,1)
        self.ActivFn=Activation
    #Node=[L,E]
    #AdjacentMatrix=[L,L]
    def __call__(self,Nodes,AdjacentMatrix):
        assert AdjacentMatrix.Data.shape[0]==AdjacentMatrix.Data.shape[1]
        def AddSelfLoopThenAverage(AdjacentMatrix):
            SquareLen=AdjacentMatrix.Data.shape[0]
            AddLoopResult=(np.ones([SquareLen,SquareLen])-np.eye(SquareLen,SquareLen))*AdjacentMatrix.Data+np.eye(SquareLen,SquareLen)
            AverageResult=AddLoopResult/np.sum(AddLoopResult,axis=1)
            AdjacentMatrix.Data=AverageResult
        AddSelfLoopThenAverage(AdjacentMatrix)
        HardAttentionResultAndSum=MatMul(AdjacentMatrix,Nodes)
        #return self.ActivFn(Add(MatMul(HardAttentionResultAndSum,self.M),self.b))
        return self.ActivFn(MatMul(HardAttentionResultAndSum,self.M))