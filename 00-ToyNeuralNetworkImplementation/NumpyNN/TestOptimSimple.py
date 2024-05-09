from Model import *
from Optim import SimpleGradOptim as SGO
from Optim import MomentumGradOptim as MGO
from Tensor import InputTensor
from  OperatorBase import GOperatorManager,GDataNodeManager
import  numpy as np
FModel=FNNModel([2,1])
LossFunc=Norm2Dis
y=FModel(InputTensor(np.array([4,5])))
#x=InputTensor(np.array([7]))
FModel.OneRound()
print(y.Data)
print(y.Grad)
print(FModel.FFN.M[0].Data)
print(FModel.FFN.M[0].Grad)
print(FModel.FFN.b[0].Data)
print(FModel.FFN.b[0].Grad)

print("xxx")
y=FModel(InputTensor(np.array([4,5])))
FModel.OneRound()
print(y.Data)
print(y.Grad)
print(FModel.FFN.M[0].Data)
print(FModel.FFN.M[0].Grad)
print(FModel.FFN.b[0].Data)
print(FModel.FFN.b[0].Grad)
