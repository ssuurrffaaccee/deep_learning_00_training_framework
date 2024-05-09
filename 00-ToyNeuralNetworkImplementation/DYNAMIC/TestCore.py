from Core import *
def TestF1():
    print("Test1")

    Node1=GradPro(3.0)
    Node2=GradPro(-2.0)
    Node3=Add().Register()(Node1,Node2)
    Node4=GradPro(-7.0)
    Node5=GradPro(4.0)
    Node6=Add().Register()(Node4,Node5)
    Node7=Add().Register()(Node3,Node6)
    Node8=Mul().Register()(Node7,Node7)
    Node9=Relu().Register()(Node8)
    
    print("---------------------Forward-----------------------------------------------------------")
    Forward()
    print("--------------------Backward------------------------------------------------------------")
    Backward()
    ToDot("dot/CoreT1.gv")
def TestF2():
    print("Test2")
    #Build a Promise
    Node1=GradPro(1.0)
    Node2=Add().Register()(Node1,Node1)
    Node3=Add().Register()(Node2,Node2)
    Node4=Add().Register()(Node3,Node3)
    print("---------------------Forward-----------------------------------------------------------")
    Forward()
    print("--------------------Backward------------------------------------------------------------")
    Backward()
    ToDot("dot/CoreT2.gv")
def TestF3():
    print("Test1")

    Node1=GradPro(3.0)
    Node2=GradPro(-2.0)
    Node3=Minus().Register()(Node1,Node2)
    Node4=GradPro(-7.0)
    Node5=GradPro(4.0)
    Node6=Minus().Register()(Node4,Node5)
    Node7=Add().Register()(Node3,Node6)
    Node8=Mul().Register()(Node7,Node7)
    Node9=Relu().Register()(Node8)
    
    print("---------------------Forward-----------------------------------------------------------")
    Forward()
    print(GOperator.Operators)
    print("--------------------Backward------------------------------------------------------------")
    Backward()
    Forward()
    print(GOperator.Operators)
    ToDot("dot/CoreT3.gv")
def TestF4():
    print("Test4")

    Node1=GradPro(3.0)
    Node2=GradPro(-2.0)
    Node3=Max().Register()(Node1,Node2)
    Node4=GradPro(-7.0)
    Node5=GradPro(4.0)
    Node6=Max().Register()(Node4,Node5)
    
    Node7=Add().Register()(Node3,Node6)
    
    Node8=Max().Register()(Node7,Node7)
    
    Node9=Relu().Register()(Node8)
    
    print("---------------------Forward-----------------------------------------------------------")
    Forward()
    #ToDot("dot/CoreT4.gv")
    print(GOperator.Operators)
    print("--------------------Backward------------------------------------------------------------")
    Backward()
    #Forward()
    print(GOperator.Operators)
    #need disabel Operator.Clear after Backward
    ToDot("dot/CoreT4.gv")
#TestF1()
#TestF2()
TestF4()