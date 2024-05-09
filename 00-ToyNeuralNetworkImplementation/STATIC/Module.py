from MatrixUtil  import Matrix,Sum,DotPro,MRelu,MatMul,ToDot,Forward,Backward,Vector


#tode: add module parameter random initializing functions
class Module:
    pass

class FNNRelu(Module):
    def __init__(self,Layers):
        for N in Layers:
            assert N!=0 
        self.Layers=Layers
        self.M=[]       
        for Index in range(len(Layers)-1):
            H=Layers[Index+1]
            W=Layers[Index]
            self.M.append(Matrix().BuildFromNative(H,W,[i+1 for i in range(H*W)]))
    def __call__(self,Input):
        assert Input.H==self.Layers[0] 
        Result=Input
        for Mat in self.M:
            Result=MRelu(MatMul(Mat,Result))
        return Result
class Simple2DCNNRelu(Module):
    def __init__(self,KernalSize):
        assert len(KernalSize)=2
        self.Kernal=Matrix().BuildFromDataNode(KernalSize[0],KernalSize[1],[i for i in range(self.KernalSize[0],self.KernalSize[1])])
    def __call__(self,Input):
        assert Input.H >= self.Kernal.H
        assert Input.W >= self.Kernal.W
        pass