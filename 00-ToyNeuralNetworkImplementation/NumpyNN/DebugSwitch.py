class Debug:
    def __init__(self):
        self.DEBUG=False
    def Set(self):
        self.DEBUG=True
    def Unset(self):
        self.DEBUG=False
    def __call__(self):
        return self.DEBUG
DEBUG=Debug()
DEBUG.Set()
DEBUG.Unset()