from Model import *
import  matplotlib.pyplot as plt
def TrainAndEvalProcessExample(L,Epoch):
    Optimizer=SGO(0.01)
    X=[i/L for i in range(L)]
    Y=[7*i*i/L for i in range(L)]
    InputX=[VectorInput(1,[i]) for i in X]
    InputY=[VectorInput(1,[i]) for i in Y]
    FModel=SumFFN([1,5,1])
    LossFunc=Norm2Dis
    TotalLoss=None
    def Build():
        PerPairLosses=[]
        for i in range(L):
            PerPairLosses.append(LossFunc(InputY[i],FModel(InputX[i])))
        TotalLoss=Sum(CatVectors(PerPairLosses))
    for i in range(Epoch):
        print("============== Epoch:"+str(i+1)+"=======================")
        Optimizer.ZeroGrad()
        Build()
        print(len(GOperator.Operators))
        print(len(GDataNode.Grads))
        Forward()

        Backward()
        print(len(GOperator.Operators))
        print(len(GDataNode.Consts))
        print(len(GDataNode.Grads))
        Optimizer.Step()
    ResultY=[]
    for In in InputX:
        Res=FModel.Eval(In)
        Forward()
        ResultY.append(Res.Data[0][0].Data)
    plt.figure()
    plt.plot(X,Y,"xb")
    plt.plot(X,ResultY,"or")
    plt.show()
if __name__=="__main__":
    TrainAndEvalProcessExample(100,2000)