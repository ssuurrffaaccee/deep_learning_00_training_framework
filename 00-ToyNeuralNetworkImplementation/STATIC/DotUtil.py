class Name:
    def __init__(self):
        self.Count=0
    def Reset(self):
        self.Count=0
    def GeName(self):
        Name= "N"+str(self.Count)
        self.Count=self.Count+1
        return Name
def GetName(DataNode,NameEnv,NameGenerator):
    try:
         return NameEnv[DataNode]
    except KeyError :
        Name=NameGenerator.GeName()
        NameEnv[DataNode]=Name
        return Name