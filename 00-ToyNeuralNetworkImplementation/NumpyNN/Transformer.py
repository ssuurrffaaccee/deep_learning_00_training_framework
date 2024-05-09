import torch
import torch.nn as nn
def Transpose(Tensor, Dimension1, Dimension2):
    return torch.transpose(Tensor, Dimension1, Dimension2)


def MatMul(TensorInput1, TensorInput2):
    return torch.matmul(TensorInput1, TensorInput2)


def Reshape(TensorInput, Shape):
    return TensorInput.reshape()


def VerticalMaskByMinusInfinity(TensorInput, BoolTensorMask):
    assert len(TensorInput.size()) == len(BoolTensorMask.size())
    assert BoolTensorMask.size()[-1] == 1
    return TensorInput.masked_fill(BoolTensorMask, -1e9)


def HorizontalMaskByZero(TensorInput, BoolTensorMask):
    assert len(TensorInput.size()) == len(BoolTensorMask.size())
    assert BoolTensorMask.size()[-2] == 1
    return TensorInput.masked_fill(BoolTensorMask, 0.0)


def Softmax(TensorInput, Dimension):
    return torch.nn.Softmax(TensorInput, Dimension)


def Sqrt(TensorInput):
    return troch.sqrt(TensorInput)


def Attention(Query, Key, Value, QMask, KMask):
    # Q=[B,H,L,E]
    # K=[B,H,L,E]
    # V=[B,H,L,E]
    LInd, EInd = 2, 3
    EmbeddingLength = torch.Tensor([Query.size()[-1]])
    return MatMul(HorizontalMaskByZero(Softmax(VerticalMaskByMinusInfinity(MatMul(Query, Transpose(Key, LInd, EInd))/Sqrt(EmbeddingLength), KMask), Dimension=0), QMask), Value)


class MultiHeadAttention(nn.Module):
    def __init__(self, HeadNum, EmbeddingSize, MaxLength):
        assert EmbeddingSize % HeadNum == 0
        self.HeadNum = HeadNum
        self.HeadEmbeddingLength = int(EmbeddingSize/HeadNum)
        self.EmbeddingSize = EmbeddingSize
        self.MaxLength = MaxLength
        self.HeadLinear = torch.nn.Parameter(torch.Tensor(
            HeadNum, self.HeadEmbeddingLength, self.HeadEmbeddingLength))
        self.OutputLiner = torch.nn.Parameter(
            troch.Tensor(EmbeddingSize, EmbeddingSize))

    def forward(self, Q, K, V, QMask, KMask):
        # Q=[B,L,E]
        # K=[B,L,E]
        # V=[B,L,E]
        Q, K, V = self.SplitAndPreprocess(Q, K, V)
        AttResult = Attention(Q, K, V, QMask, KMask)

        L, E = self.MaxLength, self.HeadEmbeddingLength
        HInd, LInd = 1, 2
        return MatMul(Reshape(Transpose(AttResult, HInd, LInd), [-1, L, E]), self.OutputLiner)

    def SplitAndPreprocess(self, Q, K, V):
        L, H, EDivH = self.MaxLength, self.HeadNum, self.HeadEmbeddingLength
        LInd, HInd = 1, 2
        Q = MatMul(
            Transpose(Reshape(Q, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        K = MatMul(
            Transpose(Reshape(K, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        V = MatMul(
            Transpose(Reshape(V, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        return Q, K, V


class PostionwiseFeedForward(nn.Module):
    def __init__(self, EmbeddingLength):
        self.MParameterList = torch.nn.ParameterList([torch.nn.Parameter(
            torch.Tensor(EmbeddingLength, EmbeddingLength) for i in range(2))])
        self.bParameterList = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(EmbeddingLength) for i in range(2))])
        self.Relu = torch.nn.Relu()

    def forward(self, Input):
        return MatMul(self.Relu(MatMul(Input, self.MParameterList[0])+self.bParameterList[0]), self.MParameterList[1])+self.bParameterList[1]


class EncoderLayer(nn.Module):
    def __init__(self, HeadNum, EmbeddingLength, MaxLength):
        self.HeadNum = HeadNum
        self.EmbeddingLength = EmbeddingLength
        self.MaxLength = MaxLength
        self.MultiHeadAttention = MultiHeadAttention(
            HeadNum, EmbeddingLength, MaxLength)
        self.PostionwiseFeedForward = PostionwiseFeedForward(EmbeddingLength)
        self.LayerNorm = torch.nn.LayerNorm([EmbeddingLength])

    def forward(self, Input, SrcMask):
        SelfAttentionResult = self.MultiHeadAttention(
            Input, Input, Input, SrcMask, SrcMask)
        LayerNormResultAfterAttention = self.LayerNorm(
            SelfAttentionResult+Input)

        PostionwiseFeedForwardResult = self.PostionwiseFeedForward(
            LayerNormResultAfterAttention)
        LayerNormResultAfterFeedForward = self.LayerNorm(
            PostionwiseFeedForwardResult+LayerNormResultAfterAttention)
        return LayerNormResultAfterAttention
class Encoder(nn.Module):
    def __init__(self,HeadNum,EmbeddingLength,MaxLength,LayerNum):
        self.LayerSequence=torch.nn.Sequential()
        for i in range(LayerNum):
            self.LayerSequence.add_module(EncoderLayer(HeadNum,EmbeddingLength,MaxLength))
    def forward(self,Input,SrcMask):
        return self.LayerSequence(Input,SrcMask)
