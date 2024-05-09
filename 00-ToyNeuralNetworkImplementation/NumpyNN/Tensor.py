import numpy as np
from DebugSwitch import DEBUG
class DataNodeManager:
    def __init__(self):
        self.DataNodes=[]

    def Registe(self,DataNode):
        self.DataNodes.append(DataNode)
        if DEBUG():
            print("##############")
            print("#####Add One DataNode to GLOBAL!")
            print("#####Num of DataNode: "+str(len(self.DataNodes)))
    def GetNeedOptimDataNode(self):
        return [DataNode for DataNode in self.DataNodes if DataNode.CanBeClear==False and DataNode.NeedGrad==True]
    def Clear(self):
        LeaveNodes=[DataNode for DataNode in self.DataNodes if DataNode.CanBeClear==False]
        self.DataNodes=LeaveNodes
GDataNodeManager=DataNodeManager()

class DataNode:
    def __init__(self,CanBeClear=True,NeedGrad=False):
        self.CanBeClear=CanBeClear
        self.NeedGrad=NeedGrad
        self.DEBUGself()
        self.RegisteToGlobal()
        self.Data=None
        self.Grad=None
    def RegisteToGlobal(self):
        #if not self.CanBeClear and not self.NeedGrad:
        GDataNodeManager.Registe(self)
    def SetData(self,Data):
        self.Data=Vertical(Data)
    def AddGrad(self):
        self.Grad=Vertical(np.zeros(self.Data.shape))
    def OnesGrad(self):
        self.Grad=Vertical(np.ones(self.Data.shape))
    def ZeroGrad(self):
        self.Grad=Vertical(np.zeros(self.Data.shape))
    def DEBUGself(self):
        if DEBUG():
            if self.CanBeClear==True:
                if self.NeedGrad==True:
                    print("==>IntemediateGrad being Created!")
                else:
                    print("==>InteMediateConstant being Created!")
            else:
                if self.NeedGrad==True:
                    print("==>Paremeter being Created!")
                else:
                    print("==>Input being Created")
class IntermediateConstTensor(DataNode):
    def __init__(self):
        super().__init__(CanBeClear=True,NeedGrad=False)

class IntermediateGradTensor(DataNode):
    def __init__(self):
        super().__init__(CanBeClear=True,NeedGrad=True)


class InputTensor(DataNode):
    def __init__(self,Data):
        super().__init__(CanBeClear=False,NeedGrad=False)
        self.SetData(Data)

class ParameterTensor(DataNode):
    def __init__(self,Data):
        super().__init__(CanBeClear=False,NeedGrad=True)
        self.SetData(Data)

def Vertical(numpyVector):
    Shape=numpyVector.shape
    if len(Shape)==1:
        return numpyVector.reshape((Shape[0],1))
    elif len(Shape)==0:
        return numpyVector.reshape((1,1))
    else:
        return numpyVector
from  math import sqrt
def GetParameter(H,W):
    #return ParameterTensor(np.random.normal(0.0,0.01,[H,W]))
    return ParameterTensor(np.random.normal(0.0,sqrt(2/(H+W)),[H,W]))
#def ShareMemory(DataNode,Range):
#    DataNode(CanBeClear=DataNode.CanBeClear,NeedGrad=DataNode.NeedGrad)
#    DataNode.SetData(DataNode.Data[Range[0]:Range[1]][Range[2]:Range[3]])