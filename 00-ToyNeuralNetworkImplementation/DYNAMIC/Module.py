from MatrixUtil  import Matrix,Sum,DotPro,MRelu,MatMul,ToDot,Forward,Backward,Vector,MMax,RZeroMaskMatrix,RZeroMaskVector,MMul
from MatrixUtil import GetSubMatrix,RMatrix,RVector,Norm2Dis,CatVectors,MAdd,GOperator,GDataNode,MSigmoid,MatrixInput,VectorInput,GTRAIN
#tode: add module parameter random initializing functions
class Module:
    pass

class FNNRelu(Module):
    def __init__(self,Layers):
        for N in Layers:
            assert N!=0 
        self.Layers=Layers
        self.M=[]
        self.b=[]       
        for Index in range(len(Layers)-1):
            H=Layers[Index+1]
            W=Layers[Index]
            self.M.append(RMatrix(H,W))
        for Index in range(len(Layers)):
            L=self.Layers[Index]
            if Index ==0:
                continue
            self.b.append(RVector(L))
    def __call__(self,Input):
        assert Input.H==self.Layers[0] 
        Result=Input
        for i  in range(len(self.M)):
            Result=MRelu(MAdd(MatMul(self.M[i],Result),self.b[i]))
            #Result=MRelu(MatMul(self.M[i],Result))
        return Result
class FNNSigmoid(Module):
    def __init__(self,Layers):
        for N in Layers:
            assert N!=0 
        self.Layers=Layers
        self.M=[]
        self.b=[]       
        for Index in range(len(Layers)-1):
            H=Layers[Index+1]
            W=Layers[Index]
            self.M.append(RMatrix(H,W))
        for Index in range(len(Layers)):
            L=self.Layers[Index]
            if Index ==0:
                continue
            self.b.append(RVector(L))
    def __call__(self,Input):
        assert Input.H==self.Layers[0] 
        Result=Input
        for i  in range(len(self.M)):
            #print("layer")
            Result=MSigmoid(MAdd(MatMul(self.M[i],Result),self.b[i]))
            #Result=MRelu(MatMul(self.M[i],Result))
        return Result
class Simple2DCNNRelu(Module):
    def __init__(self,KernalSize):
        assert len(KernalSize)==2
        self.Kernal=RMatrix(KernalSize[0],KernalSize[1])
        self.KH=KernalSize[0]
        self.KW=KernalSize[1]
    def __call__(self,Input):
        assert Input.H >= self.Kernal.H
        assert Input.W >= self.Kernal.W
        OutputH=Input.H-self.KH+1
        OutputW=Input.W-self.KW+1
        Indexs=[]
        for x in range(OutputH):
            for y in range(OutputW):
                #print((x,x+self.KH-1,y,y+self.KW-1))
                Indexs.append((x,x+self.KH-1,y,y+self.KW-1))
        Matrixs=[]
        for I in Indexs:
            Matrixs.append(GetSubMatrix(I[0],I[1],I[2],I[3],Input))
        Chan=[]
        for Matrix in Matrixs:
            Chan.append(Sum(MAdd(Matrix,self.Kernal)))
        Vec=CatVectors(Chan)            
        Vec.H=OutputH
        Vec.W=OutputW
        return Vec
class MaxPool(Module):
    def __init__(self,KernalSize):
        assert len(KernalSize)==2
        self.KH=KernalSize[0]
        self.KW=KernalSize[1]
    def __call__(self,Input):
        assert Input.H>=self.KH
        assert Input.W>=self.KW
        assert Input.H%self.KH==0
        assert Input.W%self.KW==0
        #Kernal=Matrix().BuildFromNative(Input.H,Input.W,[0.0 for i in range(Input.H*Input.W)])
        Ranges=[]
        for i in range(int(Input.H/self.KH)):
            for j in range(int(Input.W/self.KW)):
                #print([i*self.KH,i*self.KH+(self.KH-1),j*self.KW,j*self.KW+(self.KW-1)])
                Ranges.append([i*self.KH,i*self.KH+(self.KH-1),j*self.KW,j*self.KW+(self.KW-1)])
        SubMs=[]
        for R in Ranges:
            SubMs.append(MMax(GetSubMatrix(R[0],R[1],R[2],R[3],Input)))
        Result=CatVectors(SubMs)
        Result.H=int(Input.H/self.KH)
        Result.W=int(Input.W/self.KW)
        return Result
class DropOut(Module):
    def __init__(self,Prob=0.1):
        self.Prob=Prob
        self.ExpectationRatio=1-Prob
    def __call__(self,Input):
        if GTRAIN.IsTraining():
            #print("train")
            MaskM=RZeroMaskMatrix(Input.H,Input.W,self.Prob)
            return MMul(Input,MaskM)
        #can do this <-when eval do not need backward
        if not GTRAIN.IsTraining():
            #print("Eval")
            ExpectationM=Matrix().BuildFromNative(Input.H,Input.W,[self.ExpectationRatio for i in range(Input.H*Input.W)])
            return MMul(Input,ExpectationM)
    