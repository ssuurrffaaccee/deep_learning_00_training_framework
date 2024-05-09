from Module import *

X=[i/100 for i in range(100)]
Xp=[VectorInput(1,[i]) for i in X]
Y=[2*i/100 for i in range(100)]
Yp=[VectorInput(1,[i]) for i in Y]
YO=[]
F=FNNRelu([1,2,1])

def Build():
    O=[]
    for i in range(100):
        O.append(Norm2Dis(F(Xp[i]),Yp[i]))
    return Sum(CatVectors(O))

Epoch=4
for i in range(Epoch):
    #feed Operators
    R=Build()
    #FN="dot/Relu"+str(i)+".gv"
    #feed Consts
    Forward()
    #ToDot(FN)
    print(len(GDataNode.Consts))
    Backward()
    YO.append(R.Data[0][0].Data)
    print(len(GOperator.Operators))
    print(len(GDataNode.Grads))
print(F.M[0].Data[0][0].Data)
print(F.b[0].Data[0][0].Data)
print(Y)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X,Y)
plt.show()
