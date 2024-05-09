

class DataNode:
    def __init__(self):
        self.Grad=0.0
        self.bInput=False

def PrintDataNode(DataNode):
    print("---------------------")
    print("--Data:")
    print(DataNode[0].Data)
    print("--Grad:")
    print(DataNode[0].Grad)
class ConstDataNode(DataNode):
    def __init__(self,FloatNum):
        super().__init__()
        self.Const=True
        self.Data=FloatNum
    
class GradDataNode(DataNode):
    def __init__(self,FloatNum):
        super().__init__()
        self.Const=False
        self.Data=FloatNum

import DotUtil

class GlobalDataNodeManagement:
    def __init__(self):
        self.Consts=[]
        self.Grads=[]
    def CreateConstNode(self,FloatNum):
        Node=ConstDataNode(FloatNum)
        self.Consts.append(Node)
        return Node
    def CreateGradNode(self,FloatNum):
        Node=GradDataNode(FloatNum)
        self.Grads.append(Node)
        return Node
    def ClearConsts(self):
        Input=[]
        for Elem in self.Consts:
            if  Elem.bInput:
                Input.append(Elem)
        self.Consts=Input
    def ClearGrads(self):
        self.Grads=[]
GDataNode=GlobalDataNodeManagement()
def Const(FloatNum):
    return GDataNode.CreateConstNode(FloatNum)
def Grad(FloatNum):
    return GDataNode.CreateGradNode(FloatNum)
#ConstPromise
def ConstPro(FloatNum):
    return [Const(FloatNum)]
#GradPromise
def GradPro(FloatNum):
    return [Grad(FloatNum)]

class GlobalOperatorManagement:
    def __init__(self):
        self.Operators=[]
    def Forward(self):
        #clear for dynamic,when ForWard() is called, Consts will be feeded
        #delay it in after Backward.because need allocate Const in build pahse e. DropOut
        #GDataNode.ClearConsts()
        for Operator  in self.Operators:
            Operator.Forward()
    def Backward(self,Clear=True):
        self.Operators[-1].OutputPromise[0].Grad=1.0
        Indexs=list(range(len(self.Operators)))
        Indexs.reverse()
        for Index in Indexs:
            self.Operators[Index].Backward()
        if Clear:
            self.Clear()
            GDataNode.ClearConsts()
    def AddOperator(self,Operator):
        self.Operators.append(Operator)
    def Clear(self):
        self.Operators=[]

GOperator=GlobalOperatorManagement()
class TRAINSWITCH:
    def __init__(self):
        self.TRAIN=True
    def IsTraining(self):
        return self.TRAIN
    def SetTrain(self):
        self.TRAIN=True
    def SetEval(self):
        self.TRAIN=False
GTRAIN=TRAINSWITCH()


def ToDot(FileName):
    Head="digraph{\n"
    Tail="}\n"
    DefineString=""
    Env={}
    NameGen=DotUtil.Name()
    for Elem in GDataNode.Consts:
        DefineString=DefineString+DotUtil.GetName(Elem,Env,NameGen)+"[label=\""+str(Elem.Data)+"_"+str(Elem.Grad)+"\"]\n"
    for Elem in GDataNode.Grads:
        DefineString=DefineString+DotUtil.GetName(Elem,Env,NameGen)+"[label=\""+str(Elem.Data)+"_"+str(Elem.Grad)+"\"]\n"
    EdgeString=""
    for Operator in GOperator.Operators:
        for Promise in Operator.InputPromises:
            Node=Promise[0]
            EdgeString=EdgeString+DotUtil.GetName(Node,Env,NameGen)+"->"+DotUtil.GetName(Operator.OutputPromise[0],Env,NameGen)+"[label=\""+type(Operator).__name__+"\"]\n"
    with open(FileName,"w") as fd:
        fd.write(Head+DefineString+EdgeString+Tail)

class FunctionNode:
    def __init__(self):
        #[[Datanode]]
        self.InputPromises=[]
        #[DataNode]
        self.OutputPromise=["PlaceHolder"]
    def __call__(self):
        pass
    #let higer builder to registe me to GOperator
    def Register(self):
        GOperator.AddOperator(self)
        return self
    def Forward(self):
        #when the node do forward computation,it  add itself to GOperators
        self.OutputPromise.pop() 
        #write to Promise
        self.OutputPromise.append(Const(self.Calculate(self.InputPromises)))
    #[Promise]->Float
    def Calculate(self,InputPromises):
        return 1.0
    #DataNode->Float
    def LocalGrad(self,Inputs):
        return 0.0
    def Backward(self):
        DownStreamGrad=self.OutputPromise[0].Grad
        for Promise in self.InputPromises:
            Promise[0].Grad=Promise[0].Grad+DownStreamGrad*self.LocalGrad(Promise[0])

class TwoOperandFunction(FunctionNode):
    def __init__(self):
        super().__init__()
    #[DataNode],[DataNode]->[DataNode]
    def __call__(self,InputPromise1,InputPromise2):
        #this moment ,in  InputPromise,there is no DataNode instance
        self.InputPromises=[InputPromise1,InputPromise2]
        #return Promise
        return self.OutputPromise
class OneOperandFunction(FunctionNode):
    def __init__(self):
        super().__init__()
    def __call__(self,InputPromise):
        self.InputPromises=[InputPromise]
        #self.Output=GDataNode.CreateConstNode(self.Calculate(self.Inputs))
        #return Promise
        return self.OutputPromise
class Add(TwoOperandFunction):
    def __init__(self):
        super().__init__()
    def Calculate(self,InputPromises):
        #get promise
        DataNode1=InputPromises[0][0]
        DataNode2=InputPromises[1][0]
        return DataNode1.Data+DataNode2.Data
    
    def LocalGrad(self,DataNode):
        return 1.0
class Minus(TwoOperandFunction):
    def __init__(self):
        super().__init__()
    def Calculate(self,InputPromises):
        #get promise
        DataNode1=InputPromises[0][0]
        DataNode2=InputPromises[1][0]
        return DataNode1.Data-DataNode2.Data
    
    def LocalGrad(self,DataNode):
        InputPromises=self.InputPromises
        if DataNode==InputPromises[0][0]:
            return 1.0
        else:
            return -1.0
class Mul(TwoOperandFunction):
    def __init_(self):
        super().__init__()
    #[Promise]->Float
    def Calculate(self,InputPromises):
        #get promise
        DataNode1=InputPromises[0][0]
        DataNode2=InputPromises[1][0]
        return DataNode1.Data*DataNode2.Data
    
    def LocalGrad(self,DataNode):
        InputPromises=self.InputPromises
        if DataNode==InputPromises[0][0]:
            return InputPromises[1][0].Data
        else:
            return InputPromises[0][0].Data

class Max(TwoOperandFunction):
    def __init__(self):
        super().__init__()
        self.Count=0
        self.TwoInputIsSame=False
    def Calculate(self,InputPromises):
        DataNode1=InputPromises[0][0]
        DataNode2=InputPromises[1][0]
        if DataNode1==DataNode2:
            self.TwoInputIsSame=True
        if DataNode1.Data>DataNode2.Data:
            return DataNode1.Data
        else:
            return DataNode2.Data
    def LocalGrad(self,DataNode):
        DataNode1=self.InputPromises[0][0]
        DataNode2=self.InputPromises[1][0]
        if self.TwoInputIsSame:
            if self.Count==0:
                self.Count=self.Count+1
                return 1.0
            else:
                return 0.0
        else:
            if DataNode.Data>=DataNode1.Data and DataNode.Data>=DataNode2.Data:
                return 1.0
            else:
                return 0.0

class Relu(OneOperandFunction):
    def __init__(self):
        super().__init__()
    #[Promises]->float
    def Calculate(self,InputPromises):
        DataNode=InputPromises[0][0]
        return max(0,DataNode.Data)
    
    def LocalGrad(self,DataNode):
        if DataNode.Data>0:
            return 1.0
        else:
            return 0.0



class Sigmoid(OneOperandFunction):
    def __init__(self):
        super().__init__()
    #[Promises]->float
    def Calculate(self,InputPromises):
        DataNode=InputPromises[0][0]
        #return max(0,DataNode.Data)
        return 1.0/(1+pow(2.718281,-DataNode.Data))
    
    def LocalGrad(self,DataNode):
        #if DataNode.Data>0:
        #   return 1.0
        #else:
        #    return 0.0
        OuputData=self.OutputPromise[0].Data
        Grad=OuputData*(1-OuputData)
        #print("Sigmoid: "+str(Grad))
        return Grad

def Forward():
    GOperator.Forward()
def Backward(Clear=True):
    GOperator.Backward(Clear)
