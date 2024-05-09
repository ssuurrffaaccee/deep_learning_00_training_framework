from Tensor import DataNode,GDataNodeManager
from DebugSwitch import DEBUG
class Operator:
    OperatorID=0
    #public
    def __init__(self,Name=""):
        self.Inputs=[]
        self.Output=None
        Operator.OperatorID=Operator.OperatorID+1
        if Name=="":
            self.DEBUGID=str(Operator.OperatorID)
        else:
            self.DEBUGID=str(Operator.OperatorID)+" @ "+Name
    #public
    def __call__(self):
        pass
    #protected
    def Calculate(self):
        return self.Output
    #public
    def Forward(self):
        if DEBUG():
            print("----------------------")
            print(type(self).__name__+self.DEBUGID+"->Forward Calculate")
        self.Calculate()
        self.RegisteBackward()
    #private
    def RegisteBackward(self):
        if self.OutputNeedGrad():
            if DEBUG():
                print(type(self).__name__+self.DEBUGID+"->RegisteBackward")
            GOperatorManager.EnqueBackward(self)
            if DEBUG():
                print("Num of Backward: "+str(len(GOperatorManager.BackwardOperators)))
    #public
    def RegisteForward(self):
        GOperatorManager.EnqueForward(self)
        if DEBUG():
            print("-----------------------")
            print(type(self).__name__+self.DEBUGID+"->RegisterForward")
            print("Num of Backward: "+str(len(GOperatorManager.ForwardOperators)))
        return self
    #private
    def OutputNeedGrad(self):
        NeedGrad=False
        for DataNode in self.Inputs:
            if DataNode.NeedGrad==True:
                NeedGrad=True
                self.Output.NeedGrad=True
                break
        if DEBUG():
            if NeedGrad:
                print(type(self).__name__+self.DEBUGID+"->Need Backward!")
            else:
                print(type(self).__name__+self.DEBUGID+"->Do not Need Backward")
        return NeedGrad
    #public
    def Backward(self):
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Backward Calculate")
        for DataNode in self.Inputs:
            if not DataNode.NeedGrad:
                continue
            else:
                DownStreamGrad=self.Output.Grad
                if type(DataNode.Grad)==type(None):
                    DataNode.AddGrad()
                DataNode.Grad=DataNode.Grad+self.LocalGrad(DataNode,DownStreamGrad)
    #protected
    def LocalGrad(self,DataNode,DownStreamGrad):
        pass
class OneOperandOperator(Operator):
    def __init__(self,Name=""):
        super().__init__(Name)
        #self.Output=DataNode(AutoRegiste=False)
    def __call__(self,TensorInput):
        self.Inputs.append(TensorInput)
        self.Output=DataNode()
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        pass

class TwoOperandOperator(Operator):
    def __init__(self,Name=""):
        super().__init__(Name)
        #self.Output=DataNode(AutoRegiste=False)
    def __call__(self,TensorInput1,TensorInput2):
        self.Inputs.append(TensorInput1)
        self.Inputs.append(TensorInput2)
        self.Output=DataNode()
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        pass
class MultiOperandOperator(Operator):
    def __init__(self,Name=""):
        super().__init__(Name)
        #self.Output=DataNode(AutoRegiste=False)
    def __call__(self,TensorInputList):
        self.Inputs.extend(TensorInputList)
        self.Output=DataNode()
        if DEBUG():
            print(type(self).__name__+self.DEBUGID+"->Builded")
        return self.Output
    def Calculate(self):
        pass
class OperatorManager:
        def __init__(self):
            self.ForwardOperators=[]
            self.BackwardOperators=[]
        def EnqueForward(self,Operator):
            self.ForwardOperators.append(Operator)
        def EnqueBackward(self,Operator):
            self.BackwardOperators.append(Operator)
        def Forward(self):
            for Operator in self.ForwardOperators:
                Operator.Forward()
        def Backward(self):
            self.BackwardOperators[-1].Output.OnesGrad()
            if DEBUG():
                print("Last Output Grad Oneing finished!")
            self.BackwardOperators.reverse()
            for Operator in self.BackwardOperators:
                Operator.Backward()
        def Clear(self):
            self.ForwardOperators=[]
            self.BackwardOperators=[]
GOperatorManager=OperatorManager()

def Forward():
    if DEBUG():
        print("=======Forward Begining!=======")
    GOperatorManager.Forward()

def Backward():
    if DEBUG():
        print("=======Backward Begining!=======")
    GOperatorManager.Backward()

def ClearOperator():
    if DEBUG():
        print("=======Clear Operator!========")
        print("Num Of Data Befor Clear: "+str(len(GOperatorManager.ForwardOperators))+":"+str(len(GOperatorManager.BackwardOperators)))
    GOperatorManager.Clear()
    if DEBUG():
        print("Num Of Data Befor Clear: "+str(len(GOperatorManager.ForwardOperators))+":"+str(len(GOperatorManager.BackwardOperators)))

def ClearDataNode():
    if DEBUG():
        print("=======Clear DataNode!========")
        print("Num Of Data Befor Clear: "+str(len(GDataNodeManager.DataNodes)))
    GDataNodeManager.Clear()
    if DEBUG():
        print("Num Of Data After Clear: "+str(len(GDataNodeManager.DataNodes)))
class Train:
    def __init__(self):
        self.TRAIN=True
    def Train(self):
        self.TRAIN=True
    def Eval(self):
        self.TRAIN=False
    def __call__(self):
        return self.TRAIN
TRAIN=Train()
TRAIN.Train()