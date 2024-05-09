def LoadTestData():
    Images=LoadImage("minst/t10k-images-idx3-ubyte")
    Labels=LoadLable("minst/t10k-labels-idx1-ubyte")
    Dataset={}
    assert len(Images)==len(Labels)
    return list(zip(Images,Labels))
def LoadTrainData():
    Images=LoadImage("minst/train-images-idx3-ubyte")
    Labels=LoadLable("minst/train-labels-idx1-ubyte")
    Dataset={}
    assert len(Images)==len(Labels)
    return list(zip(Images,Labels))
def BytesToInt(Bytes):
    return int.from_bytes(Bytes,"big")
def LoadImage(FileName):
    Images=[]
    File=open(FileName,"rb")
    C=File.read()
    assert BytesToInt(C[:4])==2051
    ImageNum=BytesToInt(C[4:8])
    ImageH=BytesToInt(C[8:12])
    ImageW=BytesToInt(C[12:16])
    Len=ImageH*ImageW
    for ImaInd in range(ImageNum):
        LeftIncre=ImaInd*Len
        Img=C[16+LeftIncre:16+LeftIncre+Len]
        NumImg=[]
        for i in Img:
            NumImg.append(i)
        Images.append(NumImg)
    return Images
def LoadLable(FileName):
    Labels=[]
    File=open(FileName,"rb")
    C=File.read()
    assert BytesToInt(C[:4])==2049
    LabelNum=BytesToInt(C[4:8])
    for Label  in C[8:]:
            Labels.append(Label)

    return Labels
def OneHot(Num,Len):
    Hot=[0 for i in range(Len)]
    Hot[Num]=1
    return Hot
def Test():
    x=LoadTestData()
    #y=LoadTrainData()
    for z in x:
        print(z)
        break
if __name__=="__main__":
    Test()