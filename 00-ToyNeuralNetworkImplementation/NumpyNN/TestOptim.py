from Model import *
from Optim import SimpleGradOptim as SGO
from Optim import MomentumGradOptim as MGO
from Tensor import InputTensor
from  OperatorBase import GOperatorManager,GDataNodeManager
import  numpy as np
import  matplotlib.pyplot as plt
def FNNSumTrainAndEvalProcessExample(L,Epoch):
    Optimizer=SGO(0.1)
    #Optimizer=MGO(0.1)
    X=[i for i in range(L)]
    Y=[7*i*i for i in range(L)]
    InputX=[InputTensor(np.array([i])) for i in X]
    InputY=[InputTensor(np.array([i])) for i in  Y]
    FModel=FNNModel([1,1])
    LossFunc=Norm2Dis
    def Build():
        PerPairLosses=[]
        for i in range(L):
            PerPairLosses.append(LossFunc(InputY[i],FModel(InputX[i])))
        TotalLoss=Sum(Concat(PerPairLosses))
        return TotalLoss
    BatchEpoch=20
    for i in range(Epoch):
        for j in range(BatchEpoch):
            print("============== Epoch:"+str(i+1)+":"+str(j)+"=======================")
            Optimizer.ZeroGrad()
            TotalLoss=Build()
            print(len(GOperatorManager.ForwardOperators))
            print(len(GDataNodeManager.DataNodes))
            FModel.OneRound()
            Optimizer.Step()
            print(TotalLoss.Data)
            #if i==10:
            #    return
    ResultY=[]
    for In in InputX:
        Res=FModel(In,Train=False)
        FModel.Forward()
        ResultY.append(Res.Data)
    plt.figure()
    plt.plot(X,Y,"xb")
    plt.plot(X,np.concatenate(ResultY),"or")
    plt.show()
from LoadMinst import LoadTestData,LoadTrainData
def MinstFNNSumTrainAndEvalProcessExample(Epoch):
    #Optimizer=SGO(0.1)
    Optimizer=MGO(0.1)
    BatchSize=20
    TrainData=SplitByBatch(BatchSize,LoadTrainData())
    FModel=FNNMinstModel(28*28,[5,5,5],10)
    LossFunc=SoftmaxEntropy
    TotalLoss=None
    def Build(Batch):
        PerPairLosses=[]
        for i in range(len(Batch)):
            Tensor1=InputTensor(np.array(OneHot(Batch[i][1],10)))
            Tensor2=InputTensor(np.array(Batch[i][0])/500)
            PerPairLosses.append(LossFunc(FModel(Tensor2),Tensor1))
        TotalLoss=Sum(Concat(PerPairLosses))
        return TotalLoss
    for i in range(Epoch):
        print("============== Epoch:"+str(i+1)+"=======================")
        Optimizer.ZeroGrad()
        TLL=Build(TrainData[i])
        print(len(GOperatorManager.ForwardOperators))
        print(len(GDataNodeManager.DataNodes))
        FModel.Forward()
        print(len(GOperatorManager.BackwardOperators))
        FModel.Backward()
        print("TLLLLLLLLLLLLLLLL====================")
        print(TLL.Data)
        Optimizer.Step()
    ResultY=[]
    RefrenceY=[]
    TestData=LoadTestData()

    TRAIN.Eval()
    for ind,In in enumerate(TestData):
        print("Testing::"+str(ind))
        RefrenceY.append(In[1])
        Res=FModel(InputTensor(np.array(In[0])/500),False)
        FModel.Forward()
        print(Res.Data)
        ResultY.append(np.argmax(Res.Data))
        FModel.Clear()
        if ind==1000:
            break
    plt.figure()
    plt.plot(ResultY,"xb")
    plt.plot(RefrenceY,"or")
    plt.show()
def SplitByBatch(Batch,XY):
    Batchs=[]
    Indexes=[]
    Len=len(XY)
    for i in range(len(XY)):
        if i*Batch<=Len:
            Indexes.append(i*Batch)
    if Indexes[-1]!=Len:
        Indexes.append(Len)
    for j in range(len(Indexes)-1):
        Batchs.append(XY[Indexes[j]:Indexes[j+1]])
    return Batchs
def OneHot(Num,Len):
    Hot=[0 for i in range(Len)]
    Hot[Num]=1
    return Hot
def SmoothOneHot(Num,Len):
    Hot=[0.1/Len for i in range(Len)]
    Hot[Num]=Hot[Num]+0.9
    return Hot
def OneHotToNum(OneHotRepr):
    Count=0
    for x in OneHotRepr:
        if x==1:
            break
        else:
            Count=Count+1
    return Count
def MinstCNNTrainAndEvalProcessExample(Epoch):
    def SplitByBatch(Batch,XY):
        Batchs=[]
        Indexes=[]
        Len=len(XY)
        for i in range(len(XY)):
            if i*Batch<=Len:
                Indexes.append(i*Batch)
        if Indexes[-1]!=Len:
            Indexes.append(Len)
        for j in range(len(Indexes)-1):
            Batchs.append(XY[Indexes[j]:Indexes[j+1]])
        return Batchs
    Optimizer=SGO(0.1)
    #Optimizer=MGO(0.1)
    BatchSize=1
    TrainData=SplitByBatch(BatchSize,LoadTrainData())
    FModel=MinstCNNModel()
    LossFunc=Entropy
    TotalLoss=None
    def Build(Batch):
        PerPairLosses=[]
        for i in range(len(Batch)):
            Tensor1=InputTensor(np.array(SmoothOneHot(Batch[i][1],10)))
            Tensor2=InputTensor(np.array(Batch[i][0]).reshape([28,28]))
            PerPairLosses.append(LossFunc(FModel(Tensor2),Tensor1))
        TotalLoss=Sum(Concat(PerPairLosses))
        return TotalLoss
    BatchEpoch=4

    plt.figure()
    Data=[]

    for i in range(Epoch):
        for j in range(BatchEpoch):
            print("============== Epoch:"+str(i+1)+":"+str(j)+"=======================")
            Optimizer.ZeroGrad()
            TLL=Build(TrainData[i])
            print(len(GOperatorManager.ForwardOperators))
            print(len(GDataNodeManager.DataNodes))
            FModel.Forward()
            print(len(GOperatorManager.BackwardOperators))
            FModel.Backward()
            print("TLLLLLLLLLLLLLLLL====================")
            print(TLL.Data)
            Data.append(TLL.Data[0][0])
            Optimizer.Step()
        if i==1000:
            break
    plt.plot(Data)
    ResultY=[]
    RefrenceY=[]
    TestData=LoadTestData()

    TRAIN.Eval()
    for ind,In in enumerate(TestData):
        print("Testing::"+str(ind))
        RefrenceY.append(In[1])
        Res=FModel(InputTensor(np.array(In[0]).reshape([28,28])),False)
        FModel.Forward()
        print(Res.Data)
        ResultY.append(np.argmax(Res.Data))
        FModel.Clear()
        if ind==1000:
            break
    plt.figure()
    plt.plot(ResultY,"xb")
    plt.plot(RefrenceY,"or")
    plt.show()
#FNNSumTrainAndEvalProcessExample(100,2000)
#from LoadMinst import LoadTestData,LoadTrainData
#def MinstTrainAdnEval():
#    def SplitByBatch(Batch,XY):
#        Batchs=[]
#        Indexes=[]
#        Len=len(XY)
#        for i in range(len(XY)):
#            if i*Batch<=Len:
#                Indexes.append(i*Batch)
#        if Indexes[-1]!=Len:
#            Indexes.append(Len)
#        for j in range(len(Indexes)-1):
#            Batchs.append(XY[Indexes[j]:Indexes[j+1]])
#        return Batchs
#    def SwithXYToMaxtrix(XY):
#        Data=[i/255 for i in XY[0]]
#        X=MatrixInput(28,28,Data)
#        Y=VectorInput(10,[0.0 for i in range(10)])
#        Y.Data[XY[1]][0].Data=1.0
#        return X,Y
#    #print(SplitByBatch(2,[i for i in range(10)]))
#    #print(SplitByBatch(3,[i for i in range(10)]))
#    #assert 1==2
#    BATCH=4
#    Optimizer=SGO(0.01)
#    TrainXY=SplitByBatch(BATCH,LoadTrainData()[:100])
#    print(TrainXY[0][0])
#    print(len(TrainXY[0]))
#    print(len(TrainXY))
#    FModel=MinstCNN()
#    LossFunc=Norm2Dis
#    TotalLoss=None
#    def Build(OneBatchXY):
#        PerPairLosses=[]
#        for i in range(BATCH):
#            X,Y=SwithXYToMaxtrix(OneBatchXY[i])
#            PerPairLosses.append(LossFunc(Y,FModel(X)))
#        TotalLoss=Sum(CatVectors(PerPairLosses))
#        return TotalLoss
#    #assert 1==2
#    Epoch=len(TrainXY)
#    for i in range(Epoch):
#        print("============== Epoch:"+str(i+1)+"=======================")
#        Optimizer.ZeroGrad()
#        TLost=Build(TrainXY[i])
#        print(len(GOperator.Operators))
#        print(len(GDataNode.Grads))
#        Forward()
#        Backward()
#        print(len(GOperator.Operators))
#        print(len(GDataNode.Consts))
#        print(len(GDataNode.Grads))
#        Optimizer.Step()
#    ResultY=[]
#    for Elem in LoadTestData()[:100]:
#        X,_=SwithXYToMaxtrix(Elem)
#        Res=FModel.Eval(X)
#        Forward()
#        Onehot=[]
#        for i in range(10):
#           Onehot.append(Res.Data[i][0].Data)
#        ResultY.append(Onehot)
#    f=open("Result.txt","w")
#    for Elem in ResultY:
#        Line=""
#        for Da in Elem:
#            Line=Line+" "+str(Da)
#        Line=Line+"\n"
#        f.write(Line)
#    f.close()
#MinstTrainAdnEval()
#FNNSumTrainAndEvalProcessExample(100,1000)
#MinstFNNSumTrainAndEvalProcessExample(2000)
MinstCNNTrainAndEvalProcessExample(1000)