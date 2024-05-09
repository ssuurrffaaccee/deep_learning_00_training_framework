from Core import Add,Mul,Relu,GradPro,ConstPro,Forward,Backward,ToDot,Minus,GOperator,Sigmoid,GDataNode,Max,GTRAIN
from Rand import Random,RandomZeroMask
class Matrix:
    def __init__(self,Grad=False):
        super().__init__()
        self.Type="Matrix"
        if Grad:
            self.Creator=GradPro
        else:
            self.Creator=ConstPro
    def BuildFromNative(self,H,W,NativeList):
        self.H=H
        self.W=W
        self.Data=[self.Creator(i) for i in NativeList]
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
def GetSubMatrix(H1,H2,W1,W2,MatrixIn):
    assert H1<H2
    assert W1<W2
    assert MatrixIn.H>=H1 and MatrixIn.H>=H2
    assert MatrixIn.W>=W1 and MatrixIn.W>=W2
    Indexs=[]
    for x in range(H1,H2+1):
        for y in range(W1,W2+1):
            #print((x,y))
            Indexs.append((x,y))
    ProData=[]
    for I in Indexs:
        ProData.append(GetElem(I[0],I[1],MatrixIn))
    return Matrix().BuildFromDataNode(H2-H1+1,W2-W1+1,ProData)  
def RMatrix(H,W):
    #Seed()
    RandomData=[Random() for i in range(H*W)]
    return Matrix(Grad=True).BuildFromNative(H,W,RandomData)
def RVector(L):
    #Seed()
    RandomData=[Random() for i in range(L)]
    return Vector(Grad=True).BuildFromNative(L,RandomData)

def RZeroMaskMatrix(H,W,Prob):
    return Matrix().BuildFromNative(H,W,[RandomZeroMask(Prob) for i in range(H*W)])
def RZeroMaskVector(L,Prob):
    return Vector().BuildFromNative(L,[RandomZeroMask(Prob) for i in range(L)])

def MatrixInput(H,W,Data):
    M=Matrix().BuildFromNative(H,W,Data)
    for Elem in M.Data:
        Elem[0].bInput=True
    return M
def VectorInput(L,Data):
    V=Vector().BuildFromNative(L,Data)
    for Elem in V.Data:
        Elem[0].bInput=True
    return V
class Vector(Matrix):
    def __init__(self,Grad=False):
        super().__init__(Grad)
        self.Type="Vector"
    def BuildFromNative(self,L,NativeList):
        self.H=L
        self.W=1
        self.Data=[self.Creator(i) for i in NativeList]
        return self
    def BuildFromDataNode(self,L,DataNodeList):
        self.H=L
        self.W=1
        self.Data=DataNodeList
        return self
def CatVectors(VectorList):
    ProData=[]
    L=0
    for VectorElem in VectorList:
        L=L+VectorElem.H
        for ProNode in VectorElem.Data:
            ProData.append(ProNode)
    return Vector().BuildFromDataNode(L,ProData)
def Sum(VectorIn):
    ProData=VectorIn.Data
    ProResult=ProData[0]
    for Elem in ProData[1:]:
        ProResult=Add().Register()(Elem,ProResult)
    return Vector().BuildFromDataNode(1,[ProResult])
def MMax(VectorIn):
    ProData=VectorIn.Data
    ProResult=ProData[0]
    for Elem in ProData[1:]:
        ProResult=Max().Register()(Elem,ProResult)
    return Vector().BuildFromDataNode(1,[ProResult])
def DotPro(Vector1,Vector2):
    assert Vector1.H==Vector2.H
    ProInputs1=Vector1.Data
    ProInputs2=Vector2.Data
    ProMuls=[]
    for i  in range(len(ProInputs1)):
        ProMuls.append(Mul().Register()(ProInputs1[i],ProInputs2[i]))
    return Sum(Vector().BuildFromDataNode(Vector1.H,ProMuls))    
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
        Result.append(Relu().Register()(Elem))
    return Matrix().BuildFromDataNode(MatrixIn.H,MatrixIn.W,Result)
def MSigmoid(MatrixIn):
    Result=[]
    for Elem in MatrixIn.Data:
        Result.append(Sigmoid().Register()(Elem))
    return Matrix().BuildFromDataNode(MatrixIn.H,MatrixIn.W,Result)
def Norm2Dis(Vector1,Vector2):
    assert Vector1.H==Vector2.H
    ProData1=Vector1.Data
    ProData2=Vector2.Data
    
    ProSums=[]
    for i in range(Vector1.H):
        ProSums.append(Minus().Register()(ProData1[i],ProData2[i]))
    V=Vector().BuildFromDataNode(Vector1.H,ProSums)
    return DotPro(V,V)
def MAdd(Matrix1,Matrix2):
    assert Matrix1.H==Matrix1.H
    assert Matrix2.W==Matrix2.W
    ProData1=Matrix1.Data
    ProData2=Matrix2.Data
    ResultProData=[]
    for i in range(Matrix1.H*Matrix1.W):
        ResultProData.append(Add().Register()(ProData1[i],ProData2[i]))
    return Matrix().BuildFromDataNode(Matrix1.H,Matrix1.W,ResultProData)
def MMul(Matrix1,Matrix2):
    assert Matrix1.H==Matrix1.H
    assert Matrix2.W==Matrix2.W
    ProData1=Matrix1.Data
    ProData2=Matrix2.Data
    ResultProData=[]
    for i in range(Matrix1.H*Matrix1.W):
        ResultProData.append(Mul().Register()(ProData1[i],ProData2[i]))
    return Matrix().BuildFromDataNode(Matrix1.H,Matrix1.W,ResultProData)