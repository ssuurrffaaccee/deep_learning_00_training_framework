from Core import GDataNode,GradDataNode

class Optim:
    def __init__(self):
        pass
    def ZeroGrad(self):
        for Elem in GDataNode.Grads:
            Elem.Grad=0.0
    def Step(self):
        for Elem in GDataNode.Grads:
            self.Update(Elem)
    def Update(self,DataNode):
        pass
    def GetGrad(self):
        return [Elem.Grad for Elem in GDataNode.Grads]
    def GetData(self):
        return [Elem.Data for Elem in GDataNode.Grads]
class SimpleGradOptim(Optim):
    def __init__(self,StepLength):
        super().__init__()
        self.StepLength=StepLength
    def Update(self,DataNode):
        Data=DataNode.Data
        Grad=DataNode.Grad
        print("NewGrad: "+str(Grad))
        if Grad>10:
            Grad=10
        elif Grad<-10:
            Grad=-10
        print(Data)
        Data=Data-self.StepLength*Grad
        DataNode.Data=Data

        print(DataNode.Data)
class MomentumGradOptim(Optim):
    def __init__(self,StepLength):
        super().__init__()
        self.StepLength=StepLength
        self.GradAcumu={}
    def Update(self,DataNode):
        Data=DataNode.Data
        Grad=DataNode.Grad
        if Grad>10:
            Grad=10
        elif Grad<-10:
            Grad=-10
        Data=Data-self.StepLength*Grad
        DataNode.Data=Data
        print("NewGrad: "+str(Grad))