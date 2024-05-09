from Tensor import GDataNodeManager
import  numpy as np
class Optim:
    def __init__(self):
        self.DataNodeNeedOptim=None
    def ZeroGrad(self):
        self.DataNodeNeedOptim=GDataNodeManager.GetNeedOptimDataNode()
        for Elem in self.DataNodeNeedOptim:
            Elem.ZeroGrad()
    def Step(self):
        for Elem in self.DataNodeNeedOptim:
            self.Update(Elem)
    def Update(self,DataNode):
        pass
    def GetGrad(self):
        return [Elem.Grad for Elem in self.DataNodeNeedOptim]
    def GetData(self):
        return [Elem.Data for Elem in self.DataNodeNeedOptim]
class SimpleGradOptim(Optim):
    def __init__(self,StepLength):
        super().__init__()
        self.StepLength=StepLength
    def Update(self,DataNode):
        Data=DataNode.Data
        Grad=DataNode.Grad
        #print("NewGrad: "+str(Grad))
        #clip

        #print(GradNorm)
        #if GradNorm>=10:
        #    Grad=(Grad/GradNorm)*10
        #Grad[Grad>1]=1
        #Grad[Grad<-1]=-1
        #print(Data)
        GradNorm=np.linalg.norm(Grad)
        #print("Grad:")
        #print(Grad)
        #Data=Data-(self.StepLength*GradNorm)*Grad
        Data=Data-self.StepLength*Grad
        DataNode.Data=Data
        #print(DataNode.Data)
class MomentumGradOptim(Optim):
    def __init__(self,StepLength):
        super().__init__()
        self.StepLength=StepLength
        self.GradAcumu={}
    def GetAcumu(self,DataNode):
        try:
            return self.GradAcumu[DataNode]
        except:
            Cont=-1*DataNode.Grad*self.StepLength*np.ones(DataNode.Grad.shape)
            self.GradAcumu[DataNode]=Cont
            return Cont
    def SetAcumu(self,DataNode,Grad):
        self.GradAcumu[DataNode]=Grad
    def Update(self,DataNode):
        Data=DataNode.Data
        Grad=DataNode.Grad

        OldSlidAverageGrad=self.GetAcumu(DataNode)
        Memory=0.9
        Decay=np.sqrt(Grad*Grad+1e-7)
        NewSlidAverageGrad=OldSlidAverageGrad*Memory-Grad*self.StepLength/Decay
        self.SetAcumu(DataNode,NewSlidAverageGrad)

        Grad=NewSlidAverageGrad
        #print(Grad)
        #print("NewGrad: "+str(Grad))
        #Grad[Grad>1]=1
        #Grad[Grad<-1]=-1
        #print(Grad)

        #GradNorm=np.linalg.norm(Grad)
        #print("Grad:")
        #print(Grad)
        Data=Data+Grad
        DataNode.Data=Data
        #print(Data)