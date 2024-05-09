from Operators import OAdd,OMatMul,OMinus,OMul,ORelu,OSelect,OSum,OConcat,OTranspose,OSigmoid,ODropout,TRAIN,OSoftmaxEntropy,OSoftmaxForEval,OMaxPool,OFlatten,OSoftmax,OEntropy,OTanh
#from Tensor import IntermediateGradTensor
def Add(TensorInput1,TensorInput2,Name=""):
    return OAdd(Name).RegisteForward()(TensorInput1,TensorInput2)

def MatMul(TensorInput1,TensorInput2,Name=""):
    return OMatMul(Name).RegisteForward()(TensorInput1,TensorInput2)

def Minus(TensorInput1,TensorInput2,Name=""):
    return OMinus(Name).RegisteForward()(TensorInput1,TensorInput2)

def Relu(TensorInput,Name=""):
    return ORelu(Name).RegisteForward()(TensorInput)

def Sigmoid(TensorInput,Name=""):
    return OSigmoid(Name).RegisteForward()(TensorInput)

def Dropout(TensorInput,Dropout=0.1,Name=""):
    return ODropout(Dropout,Name).RegisteForward()(TensorInput)

def Select(TensorInput,Range,Name=""):
    return OSelect(Name).RegisteForward()(TensorInput,Range)

def Mul(TensorInput1,TensorInput2,Name=""):
    return OMul(Name).RegisteForward()(TensorInput1,TensorInput2)

def Sum(TensorInput,Name=""):
    return OSum(Name).RegisteForward()(TensorInput)

def Concat(TensorList,Shape=None,Name=""):
    assert  type(TensorList)==list
    return OConcat(Name).RegisteForward()(TensorList,Shape)
def Trans(TensorInput,Name=""):
    return OTranspose(Name).RegisteForward()(TensorInput)

def Norm2Dis(TensorInput1,TensorInput2,Name=""):
    MinuResu=OMinus(Name).RegisteForward()(TensorInput1,TensorInput2)
    TranMinuResu=OTranspose(Name).RegisteForward()(MinuResu)
    MatResu=OMatMul(Name).RegisteForward()(TranMinuResu,MinuResu)
    return OSum(Name).RegisteForward()(MatResu)
def SoftmaxForEval(TensorInput,Name=""):
    return OSoftmaxForEval(Name).RegisteForward()(TensorInput)
def SoftmaxEntropy(TensorInput1,TensorInput2,Name=""):
    return OSoftmaxEntropy(Name).RegisteForward()(TensorInput1,TensorInput2)
def MaxPool(TensorInput1,KernelSize,Name=""):
    return OMaxPool(Name).RegisteForward()(TensorInput1,KernelSize)
def Flatten(TensorInput,Name=""):
    return OFlatten(Name).RegisteForward()(TensorInput)
def Softmax(TensorInput,Name=""):
    return OSoftmax(Name).RegisteForward()(TensorInput)
def Entropy(TensorInput1,TensorInput2,Name=""):
    return OEntropy(Name).RegisteForward()(TensorInput1,TensorInput2)
def Tanh(TensorInput,Name=""):
    return OTanh(Name).RegisteForward()(TensorInput)