from Optim import SimpleGradOptim as SGO
from Module import *
from Rand import Random
import matplotlib.pyplot as plt


#plt.figure()

Xn=[i/10 for i in range(100)]
Yn=[4*i/10 for i in range(100)]
#plt.plot(Xn,Yn,'ob')
def PrepareData(Xn,Yn):
    X=[VectorInput(1,[i]) for i in Xn]
    Y=[VectorInput(1,[i]) for i in Yn]
    return list(zip(X,Y))
FNN=FNNRelu([1,1])

Data=PrepareData(Xn,Yn)
EpochResults=[]

#随机梯度下降
def Build():
    Losses=[]
    for Pair in Data:
        #print("Build>>>>")
        R=FNN(Pair[0])
        EpochResults.append(R)
        Losses.append(Norm2Dis(R,Pair[1]))
    return Sum(CatVectors(Losses))
    #ToDot("dot/Big.gv")



print(GOperator.Operators)

Epoch=2000
Opt=SGO(0.01)
for i in range(Epoch):
    print("Epoch:"+str(i))
    print(len(GOperator.Operators))
    print(len(GDataNode.Consts))
    print(len(GDataNode.Grads))
    TotalLoss=Build()
    print(len(GOperator.Operators))
    print(len(GDataNode.Consts))
    print(len(GDataNode.Grads))
    Forward()
    print(len(GOperator.Operators))
    print(len(GDataNode.Consts))
    print(len(GDataNode.Grads))
    TempY=[int(Elem.Data[0][0].Data) for Elem in EpochResults]
    String="dot/MM.gv"
    if i==0:
        ToDot(String)
    Backward()
    print(len(GOperator.Operators))
    print(len(GDataNode.Consts))
    print(len(GDataNode.Grads))
    Opt.Step()
    print(len(GOperator.Operators))
    print(len(GDataNode.Consts))
    print(len(GDataNode.Grads))
    #plt.figure()
    #plt.plot(Opt.GetGrad())
    #plt.plot(Opt.GetData())
    Opt.ZeroGrad()
    print(len(GOperator.Operators))
    #print(TotalLoss.Data[0][0].Data)

TestX=[Vector().BuildFromNative(1,[i]) for i in Xn]
OutputYn=[]
for Index, Elem in enumerate(TestX):
    #print("Test:"+str(Index))
    Result=FNN(Elem)
    Forward()
    Resultn=Result.Data[0][0].Data
    #print(Resultn)
    OutputYn.append(Resultn)
plt.figure()
plt.plot(Xn,OutputYn,"or")
plt.plot(Xn,Yn,"xb")
plt.show()


def SimpleTest():
    pass
