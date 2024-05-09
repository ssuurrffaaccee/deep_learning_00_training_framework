# Learning Neural NetWork Training Framework

## 00-ToyNeuralNetworkImplementation(Feb 29, 2020)
This extremely rudimentary training framework was implemented in Python language during my graduate studies, in order to learn the principles of neural network training. It has not been completed.
### STATIC
  - Add Tracing

    ![Add Tracing](./00-ToyNeuralNetworkImplementation/STATIC/dot/CoreT2.png)
  
  - Matrix Tracing
  
    ![Matrix Tracing1](./00-ToyNeuralNetworkImplementation/STATIC/dot/MatrixT3.png)
    ![Matrix Tracing2](./00-ToyNeuralNetworkImplementation/STATIC/dot/MatrixT4.png)
  
  - FNN Tracing
  
    ![FNN Tracing](./00-ToyNeuralNetworkImplementation/STATIC/dot/FNN.png)

### DYNAMIC
  - Mul\Add\Minus Tracing

   ![Mul\Add\Minus Tracing](./00-ToyNeuralNetworkImplementation/DYNAMIC/dot/CoreT3.png)

  - Dropout Tracing

   ![Dropout Tracing](./00-ToyNeuralNetworkImplementation/DYNAMIC/dot/DropOut.png)

  - MaxPool Tracing

   ![MaxPool Tracin](./00-ToyNeuralNetworkImplementation/DYNAMIC/dot/MaxPool.png)

  - MaxPool Tracing

   ![MaxPool Tracing](./00-ToyNeuralNetworkImplementation/DYNAMIC/dot/MaxPool.png)

  - Relu Tracing 
   
   ![Relu Tracing](./00-ToyNeuralNetworkImplementation/DYNAMIC/dot/Relu1.png)

### NumpyNN
   It has not been completed.

## 01-floatflow(November 10, 2021)
   Minimalist demo for implementing parameter servers

   Understand the training framework of deep learning neural networks through approximately 300 lines of C++ code

   ### test_linearEquation
    Solve equation 0.3*x + 0.4 = 0.8

   ## test_equation
    Solve equation 0.3*x^2 + 0.4*x + 0.5 = 0.8 

   ## test_linearFit2
    Linear fitting  a*x + b = y
    
## 02-floatflow_rs(Apr 7, 2023)
   floatflow implemented with Rust language, in order to taste Rust

## 03-dag_executor
   Used to understand the following questions:

      1. How to automatically construct a computational graph for backward calculation of backpropagation algorithm?
      
      2. How to automatically explore the parallelism of computational graphs?

      3. How can computational graphs be viewed as directed acyclic graphs to schedule the execution of computational graphs?

   - Vector self multiplication 
      ![Vector self multiplication ](./03-dagExecutor/graph/bin_VectorSelfSum.jpg)
   - Matrix-Vector multiplication
      ![Matrix-Vector multiplication](./03-dagExecutor/graph/bin_MatrixMulVector.jpg)
   - FNN 
      ![FNN](./03-dagExecutor/graph/bin_FNN.jpg)
   - High level view of Autoregressive a.k.a RNN
      ![High level view of Autoregressive a.k.a RNN](./03-dagExecutor/graph/bin_AutoregressiveH.jpg)
   - Detailed view of Autoregressive a.k. RNN
      ![Detailed view of Autoregressive a.k. RNN](./03-dagExecutor/graph/bin_Autoregressive.jpg)