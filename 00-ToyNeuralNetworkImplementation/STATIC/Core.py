

class DataNode:
    def __init__(self):
        self.Grad=0.0

def PrintDataNode(DataNode):
    print("---------------------")
    print("--Data:")
    print(DataNode.Data)
    print("--Grad:")
    print(DataNode.Grad)
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
        self.Consts=[]
    def ClearGrads(self):
        self.Grads=[]
    def Print(self):
        print("--------------------------------GDataNode:begin-------")
        print("Grad")
        for Elem in self.Grads:
            PrintDataNode(Elem)
        print("Consts")
        for Elem in self.Consts:
            PrintDataNode(Elem)
        print("--------------------------------GDataNode:end-------")

GDataNode=GlobalDataNodeManagement()
def Const(FloatNum):
    return GDataNode.CreateConstNode(FloatNum)
def Grad(FloatNum):
    return GDataNode.CreateGradNode(FloatNum)



class GlobalOperatorManagement:
    def __init__(self):
        self.Operators=[]
    def Forward(self):
        for Operator  in self.Operators:
            Operator.Forward()
    def Backward(self):
        self.Operators[-1].Output.Grad=1.0
        Indexs=list(range(len(self.Operators)))
        Indexs.reverse()
        for Index in Indexs:
            self.Operators[Index].Backward()
    def AddOperator(self,Operator):
        self.Operators.append(Operator)
    def Clear(self):
        self.Operators=[]

GOperator=GlobalOperatorManagement()



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
        for Node in Operator.Inputs:
            EdgeString=EdgeString+DotUtil.GetName(Operator.Output,Env,NameGen)+"->"+DotUtil.GetName(Node,Env,NameGen)+"[label=\""+type(Operator).__name__+"\"]\n"
    with open(FileName,"w") as fd:
        fd.write(Head+DefineString+EdgeString+Tail)

class FunctionNode:
    def __init__(self):
        self.Inputs=None
        self.Output=None
        GOperator.AddOperator(self)
    def __call__(self):
        pass
    def Forward(self): 
        self.Output.Data=self.Calculate(self.Inputs)
        return self.Output
    def Calculate(self,Inputs):
        return 1.0
    def LocalGrad(self,Inputs):
        return 0.0
    def Backward(self):
        DownStreamGrad=self.Output.Grad
        for Node in self.Inputs:
            Node.Grad=Node.Grad+DownStreamGrad*self.LocalGrad(Node)

def PrintFunctionNode(FunctionNode):
    print("-------------------")
    print(FunctionNode.Output)

class TwoOperandFunction(FunctionNode):
    def __init__(self):
        super().__init__()
    def __call__(self,DataNode1,DataNode2):
        self.Inputs=[DataNode1,DataNode2]
        self.Output=GDataNode.CreateConstNode(self.Calculate(self.Inputs))
        return self.Output
class OneOperandFunction(FunctionNode):
    def __init__(self):
        super().__init__()
    def __call__(self,DataNode):
        self.Inputs=[DataNode]
        self.Output=GDataNode.CreateConstNode(self.Calculate(self.Inputs))
        return self.Output
class Add(TwoOperandFunction):
    def __init__(self):
        super().__init__()
    def Calculate(self,Inputs):
        DataNode1=Inputs[0]
        DataNode2=Inputs[1]
        return DataNode1.Data+DataNode2.Data
    
    def LocalGrad(self,DataNode):
        return 1.0

class Mul(TwoOperandFunction):
    def __init_(self,DataNode1,DataNode2):
        super().__init__(DataNode1,DataNode2)
    def Calculate(self,Inputs):
        DataNode1=Inputs[0]
        DataNode2=Inputs[1]
        return DataNode1.Data*DataNode2.Data
    
    def LocalGrad(self,DataNode):
        Inputs=self.Inputs
        if DataNode==Inputs[0]:
            return Inputs[1].Data
        else:
            return Inputs[0].Data


class Relu(OneOperandFunction):
    def __init__(self):
        super().__init__()
    def Calculate(self,Inputs):
        DataNode=Inputs[0]
        return max(0,DataNode.Data)
    
    def LocalGrad(self,DataNode):
        if DataNode.Data>0:
            return 1.0
        else:
            return 0.0





def Forward():
    GDataNode.Print()
    GOperator.Forward()
    GDataNode.Print()
def Backward():
    GOperator.Backward()
    GDataNode.Print()
