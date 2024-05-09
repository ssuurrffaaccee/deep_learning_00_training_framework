import numpy as np
from OperatorBase import *
class OAdd(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        #print(TensorInput1.Data.shape)
        #print(TensorInput2.Data.shape)
        assert TensorInput1.Data.shape==TensorInput2.Data.shape
        AddResult=TensorInput1.Data+TensorInput2.Data
        self.Output.SetData(AddResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        return np.ones(DataNode.Data.shape)*DownStreamGrad
class OMul(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        assert TensorInput1.Data.shape==TensorInput2.Data.shape
        AddResult=TensorInput1.Data*TensorInput2.Data
        self.Output.SetData(AddResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        if DataNode==TensorInput1:
            return TensorInput2.Data*DownStreamGrad
        else:
            return TensorInput1.Data*DownStreamGrad
class OMinus(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        assert TensorInput1.Data.shape==TensorInput2.Data.shape
        AddResult=TensorInput1.Data-TensorInput2.Data
        self.Output.SetData(AddResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        if DataNode==TensorInput1:
            return np.ones(DataNode.Data.shape)*DownStreamGrad
        else:
            return -1*np.ones(DataNode.Data.shape)*DownStreamGrad
class OMatMul(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        assert TensorInput1.Data.shape[1]==TensorInput2.Data.shape[0]
        MatMulResult=np.matmul(TensorInput1.Data,TensorInput2.Data)
        self.Output.SetData(MatMulResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        if DataNode==TensorInput1:
            return np.matmul(DownStreamGrad,TensorInput2.Data.T)
        else:
            return np.matmul(TensorInput1.Data.T,DownStreamGrad)
class ORelu(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Cache=None
    def Calculate(self):
        TensorInput=self.Inputs[0]
        self.Cache=TensorInput.Data>0
        ReluResult=TensorInput.Data*self.Cache
        self.Output.SetData(ReluResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        return DownStreamGrad*self.Cache
class OSigmoid(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput=self.Inputs[0]
        SigmoidResult=1/(1+np.exp(-1*TensorInput.Data))
        self.Output.SetData(SigmoidResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        OutputData=self.Output.Data
        return DownStreamGrad*(OutputData*(1-OutputData))
class ODropout(OneOperandOperator):
    def __init__(self,Dropout=0.1,Name=""):
        super().__init__(Name)
        self.Cache=None
        self.Dropout=Dropout
        assert 0<=Dropout<1.0
    def Calculate(self):
        TensorInput=self.Inputs[0]
        if TRAIN():
            self.Cache=np.random.binomial(1,1-self.Dropout,TensorInput.Data.shape)
            DropoutResult=TensorInput.Data*self.Cache
            self.Output.SetData(DropoutResult)
        else:
            DropoutResult=TensorInput.Data*(1-self.Dropout)
            self.Output.SetData(DropoutResult)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        assert type(self.Cache)!=type(None)
        return DownStreamGrad*self.Cache
class OSum(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput=self.Inputs[0]
        #self.Cache=TensorInput.Data>0
        ReluResult=np.sum(TensorInput.Data)
        self.Output.SetData(ReluResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        return np.ones(TensorInput1.Data.shape)*DownStreamGrad
class OSelect(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def __call__(self,TensorInput,Range):
        self.Inputs.append(TensorInput)
        self.Output=DataNode()
        self.Range=Range
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        TensorInput=self.Inputs[0]
        Range=self.Range
        InputH=TensorInput.Data.shape[0]
        InputW=TensorInput.Data.shape[1]
        assert 0<=Range[0]<=Range[1]<=InputH and 0<=Range[2]<=Range[3]<=InputW
        SelectResult=TensorInput.Data[Range[0]:Range[1],Range[2]:Range[3]]
        self.Output.SetData(SelectResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput=self.Inputs[0]
        ResultGrad=np.zeros(TensorInput.Data.shape)
        ResultGrad[self.Range[0]:self.Range[1],self.Range[2]:self.Range[3]]=np.ones(DownStreamGrad.shape)*DownStreamGrad
        return ResultGrad
class OConcat(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def __call__(self,TensorList,Shape=None):
        self.Inputs.extend(TensorList)
        self.Output=DataNode()
        self.Shape=Shape
        self.ConcatShape=None
        self.Range=None
        self.FirstDimRangeDict=None
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        self.CheckAndRecordRange()
        ConcatResult=np.concatenate([DataNode.Data for DataNode in self.Inputs])
        self.ConcatShape=ConcatResult.shape
        if self.Shape!=None:
            self.ConcatShape=ConcatResult.shape
            ConcatResult=ConcatResult.reshape(self.Shape)
        self.Output.SetData(ConcatResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        if type(self.Shape)!=type(None):
            ConcatDownStreamGrad=DownStreamGrad.reshape(self.ConcatShape)
        else:
            ConcatDownStreamGrad=DownStreamGrad
        Range=self.FirstDimRangeDict[DataNode]

        return np.ones(Range[0])*ConcatDownStreamGrad[Range[1]:Range[2]]
        
    def CheckAndRecordRange(self):
        #check
        InputShapes=[DataNode.Data.shape[1:] for DataNode in self.Inputs]
        CurrentShape=InputShapes[0]
        LenOfShape=len(CurrentShape)
        ResultBool=True
        for Elem in InputShapes:
            if len(Elem) ==0:
                continue
            ResultBool=ResultBool and np.sum(Elem==CurrentShape)==LenOfShape
        assert ResultBool
        
        #record
        FirstDimRange={}
        Acumulate=0
        #print([DataNode.Data.shape for DataNode in self.Inputs])
        #print([DataNode.Data for DataNode in self.Inputs])
        for DataNode,Elem in [( DataNode , DataNode.Data.shape[0]) for DataNode in self.Inputs]:
            FirstDimRange[DataNode]=(DataNode.Data.shape,Acumulate,Acumulate+Elem)
            Acumulate=Acumulate+Elem
        self.FirstDimRangeDict=FirstDimRange
class OTranspose(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Cache=None
    def Calculate(self):
        TensorInput=self.Inputs[0]
        TransInputData=np.transpose(TensorInput.Data)
        TransResult=TransInputData*np.ones(TransInputData.shape)
        self.Output.SetData(TransResult)
    def LocalGrad(self,DataNode,DownStreamGrad):
        TransDownStreamGrad=np.transpose(DownStreamGrad)
        return TransDownStreamGrad*np.ones(TransDownStreamGrad.shape)
class OSoftmaxEntropy(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Label=None
        self.SoftmaxResult=None
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        assert TensorInput1.Data.shape[1]==1
        assert TensorInput2.Data.shape[1]==1
        assert TensorInput1.Data.shape[0]==TensorInput2.Data.shape[0]
        assert TensorInput2.CanBeClear==False
        assert TensorInput2.NeedGrad==False
        #assumption:TesorInput2 is an onehot
        #x=np.exp(TensorInput1.Data)
        #print(TensorInput1.Data)
        #print(x)
        SoftmaxResult=np.exp(TensorInput1.Data)/(np.sum(np.exp(TensorInput1.Data))+1e-9)

        Elem=SoftmaxResult[0]
        Flag=True
        for E in SoftmaxResult:
            Flag=Flag and Elem==E
        if Flag:
            SoftmaxResult=np.ones(SoftmaxResult.shape)*(1/SoftmaxResult.shape[0])
        Label=np.argmax(TensorInput2.Data)
        self.Label=Label
        self.SoftmaxResult=SoftmaxResult
        SoftmaxEntropyResult=-1*np.log(SoftmaxResult[Label])
        self.Output.SetData(SoftmaxEntropyResult)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        assert DownStreamGrad.shape[0]==1
        assert DownStreamGrad.shape[1]==1
        if DataNode==TensorInput1:
            Local=np.zeros(TensorInput1.Data.shape)
            Local[self.Label]=1-self.SoftmaxResult[self.Label]
            #print(Local*DownStreamGrad)
            return Local*DownStreamGrad
        else:
            assert "TensorInput2 should not need  Grad"
        #return DownStreamGrad*self.Cache
class OSoftmaxForEval(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput=self.Inputs[0]
        SoftmaxResult=np.exp(TensorInput.Data)/np.sum(np.exp(TensorInput.Data))
        self.Output.SetData(SoftmaxResult)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        assert "this operator is only for eval"==0
        #TensorInput1=self.Inputs[0]
        #pass
        #return DownStreamGrad*self.Cache
class OSoftmax(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Sum=None
        self.Softmax=None
    def Calculate(self):
        TensorInput=self.Inputs[0]
        #print(TensorInput.Data)
        self.Sum=np.sum(np.exp(TensorInput.Data))
        SoftmaxResult=np.exp(TensorInput.Data)/(self.Sum+1e-9)
        self.Softmax=SoftmaxResult
        self.Output.SetData(SoftmaxResult)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput=self.Inputs[0]
        return DownStreamGrad*(self.Sum-self.Softmax)*self.Softmax/(self.Sum*self.Sum+1e-9)
        #TensorInput1=self.Inputs[0]
        #pass
        #return DownStreamGrad*self.Cache
class OEntropy(TwoOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def Calculate(self):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        assert TensorInput1.Data.shape[0]==TensorInput2.Data.shape[0]
        assert TensorInput1.Data.shape[1]==1
        assert TensorInput2.Data.shape[1]==1
        Labels=TensorInput2
        Result=-1*Labels.Data*np.log(TensorInput1.Data+1e-9)
        self.Output.SetData(np.sum(Result))
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput1=self.Inputs[0]
        TensorInput2=self.Inputs[1]
        if DataNode==TensorInput2:
            return -1*DownStreamGrad*np.log(TensorInput1.Data+1e-9)
        else:
            return -1*DownStreamGrad*TensorInput2.Data*(1/(TensorInput1.Data+1e-9))
class OMaxPool(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
    def __call__(self,TensorInput,KernelSize):
        self.Inputs.append(TensorInput)
        self.Output=DataNode()
        self.KernelSize=KernelSize
        self.Indexs=None
        self.OutputH=None
        self.OutputW=None
        self.TrueIndexs=None
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        TensorInput=self.Inputs[0]
        Indexs=self.GenSubIndx()
        Result=[]
        SubIndexs=[]
        for Ind in Indexs:
            SubMatrix=TensorInput.Data[Ind[0]:Ind[1],Ind[2]:Ind[3]]
            SubIndexs.append(self.GetMaxInd(SubMatrix))
            Result.append(np.max(SubMatrix))
        Result=np.array(Result)
        Result=Result.reshape((self.OutputH,self.OutputW))
        self.Output.SetData(Result)
        self.BackToInputShape(self.Indexs,SubIndexs)
    def GenSubIndx(self):
        TensorInput=self.Inputs[0]
        InputSize=TensorInput.Data.shape
        KernelSize=self.KernelSize
        assert InputSize[0]%KernelSize[0]==0
        assert InputSize[1]%KernelSize[1]==0
        self.OutputH=int(InputSize[0]/KernelSize[0])
        self.OutputW=int(InputSize[1]/KernelSize[1])
        Indexs=[]
        for x in range(self.OutputH):
            for y in range(self.OutputW):
                Indexs.append((x*KernelSize[0],(x+1)*KernelSize[0],y*KernelSize[1],(y+1)*KernelSize[1]))
        self.Indexs=Indexs
        return Indexs
    def GetMaxInd(self,Matrix):
        x=np.argmax(np.max(Matrix,axis=1))
        y=np.argmax(np.max(Matrix,axis=0))
        return (x,y)
    def BackToInputShape(self,Indexs,SubIndexs):
        def LocalBack(Index,SubIndex):
            return (Index[0]+SubIndex[0],Index[2]+SubIndex[1])
        TrueIndexs=[]
        for I,SI in zip(Indexs,SubIndexs):
            TrueIndexs.append(LocalBack(I,SI))
        self.TrueIndexs=TrueIndexs
    def LocalGrad(self,DataNode,DownStreamGrad):
        TensorInput=self.Inputs[0]
        Result=np.zeros(TensorInput.Data.shape)
        FlattenDownStreamGrad=DownStreamGrad.reshape([self.OutputH*self.OutputW])
        for i,Ind in enumerate(self.TrueIndexs):
            Result[Ind[0]][Ind[1]]=FlattenDownStreamGrad[i]
        ResultGrad=Result
        return ResultGrad
class OFlatten(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Length=None
        self.InputH=None
        self.InputW=None
    def Calculate(self):
        TensorInput=self.Inputs[0]
        self.InputH=TensorInput.Data.shape[0]
        self.InputW=TensorInput.Data.shape[1]
        self.Length=self.InputH*self.InputW
        FlattenResult=np.ones([self.Length,1])*TensorInput.Data.reshape([self.Length,1])
        self.Output.SetData(FlattenResult)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        return DownStreamGrad.reshape(self.InputH,self.InputW)*np.ones([self.InputH,self.InputW])
class OTanh(OneOperandOperator):
    def __init__(self,Name=""):
        super().__init__(Name)
        self.Result=None
    def Calculate(self):
        TensorInput=self.Inputs[0]
        EXPPositive=np.exp(TensorInput.Data)
        EXPNegatvie=np.exp(-1*TensorInput.Data)
        Result=(EXPPositive-EXPNegatvie)/(EXPPositive+EXPNegatvie)
        self.Result=Result
        self.Output.SetData(Result)           
    def LocalGrad(self,DataNode,DownStreamGrad):
        return DownStreamGrad*(1-self.Result*self.Result)

#class ONormal(OneOperandOperator):
#    def __init__(self,Name=""):
#        super().__init__(Name)
#        self.Norm=None
#    def Calculate(self):
#        TensorInput=self.Inputs[0]
#        Norm=np.linalg.norm(TensorInput.Data)
#        self.Norm=Norm
#        NormResult=TensorInput.Data/Norm
#        self.Output.SetData(NormResult)           
#    def LocalGrad(self,DataNode,DownStreamGrad):
#        TensorInput=self.Inputs[0]
#
#        return DownStreamGrad*(1/self.Norm)