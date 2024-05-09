from Core import Add,Mul,Relu,Grad,Forward,Backward,ToDot

class Matrix:
    def __init__(self):
        super().__init__()
        self.Type="Matrix"
    def BuildFromNative(self,H,W,NativeList):
        self.H=H
        self.W=W
        self.Data=[Grad(i) for i in NativeList]
        return self
    def BuildFromDataNode(self,H,W,DataNodeList):
        self.H=H
        self.W=W
        self.Data=DataNodeList
        return self
    def SetValue(self,Values):
        assert len(Values)==self.H*self.W
        for i in range(self.H*self.W):
            self.Data[i].Data=Values[i]
def GetElem(HIndex,WIndex,Matrix):
    InnerHIndex=HIndex+1
    InnerWIndex=WIndex+1
    Index=((InnerHIndex-1)*Matrix.W+InnerWIndex)-1
    return Matrix.Data[Index]

class Vector(Matrix):
    def __init__(self):
        super().__init__()
        self.Type="Vector"
    def BuildFromNative(self,L,NativeList):
        self.H=L
        self.W=1
        self.Data=[Grad(i) for i in NativeList]
        return self
    def BuildFromDataNode(self,L,DataNodeList):
        self.H=L
        self.W=1
        self.Data=DataNodeList
        return self

def Sum(VectorIn):
    Data=VectorIn.Data
    Result=Data[0]
    for Elem in Data[1:]:
        Result=Add()(Elem,Result)
    return Vector().BuildFromDataNode(1,[Result])
def DotPro(Vector1,Vector2):
    assert Vector1.H==Vector2.H
    Inputs1=Vector1.Data
    Inputs2=Vector2.Data
    Muls=[]
    for i  in range(len(Inputs1)):
        Muls.append(Mul()(Inputs1[i],Inputs2[i]))
    return Sum(Vector().BuildFromDataNode(Vector1.H,Muls))    
def MatMul(Matrix1,Matrix2):
    def MulInIndex(HIndex,WIndex,Matrix1,Matrix2):
        Row=[]
        for i in range(Matrix1.W):
            Row.append(GetElem(HIndex,i,Matrix1))
        Column=[]
        for i in range(Matrix2.H):
            Column.append(GetElem(i,WIndex,Matrix2))
        RowVector=Vector().BuildFromDataNode(Matrix1.W,Row)
        ColumnVector=Vector().BuildFromDataNode(Matrix2.H,Column)
        return DotPro(RowVector,ColumnVector).Data[0]
    Matrix1H=Matrix1.H
    Matrix1W=Matrix1.W
    Matrix2H=Matrix2.H
    Matrix2W=Matrix2.W
    assert Matrix1W==Matrix2H
    ResultH=Matrix1H
    ResultW=Matrix2W
    ResultData=[]
    for HIndex in range(ResultH):
        for WIndex in range(ResultW):
            ResultData.append(MulInIndex(HIndex,WIndex,Matrix1,Matrix2))
    return Matrix().BuildFromDataNode(ResultH,ResultW,ResultData)
def MRelu(MatrixIn):
    Result=[]
    for Elem in MatrixIn.Data:
        Result.append(Relu()(Elem))
    return Matrix().BuildFromDataNode(MatrixIn.H,MatrixIn.W,Result)
