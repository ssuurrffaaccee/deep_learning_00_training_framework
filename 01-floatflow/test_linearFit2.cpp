#include "floatflow.h"
#include <iostream>
#include <random>

DValue *sumVector(std::vector<DValue *> vs, Tape *t) {
  DValue *sum = constant(0);
  for (auto *v : vs) {
    sum = add(sum, v, t);
  }
  return sum;
}
int main() {
  ParameterServer ps = ParameterServer();
  ps.regist(ParameterServer::Value{
      "a", "sgd", {{"lr", 0.01}, {"beta0", 0.9}, {"beta1", 0.99}}, 3});
  ps.regist(ParameterServer::Value{
      "b", "sgd", {{"lr", 0.01}, {"beta0", 0.9}, {"beta1", 0.99}}, 2});
  DValue *a = parameter();
  DValue *b = parameter();
  int batchSize = 32;
  std::vector<DValue *> batchedX, batchedY, batchedLoss;
  Tape t = Tape();
  // a*x + b = y
  for (int i = 0; i < batchSize; i++) {
    DValue *x = placeholder();
    DValue *z = linear(a, x, b, &t);
    DValue *y = placeholder();
    DValue *loss = L2(z, y, &t);
    batchedX.push_back(x);
    batchedY.push_back(y);
    batchedLoss.push_back(loss);
  }
  DValue *batchLoss = sumVector(batchedLoss, &t);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (int i = 0; i < 100000; i++) {
    a->v = ps.pull("a");
    b->v = ps.pull("b");
    for (int j = 0; j < batchSize; j++) {
      float xSample = dis(gen);
      float ySample = 0.4 * xSample + 0.6;
      batchedX[j]->v = xSample;
      batchedY[j]->v = ySample;
    }
    t.forward();
    batchLoss->dv = 1.0f / batchSize;
    t.backward();
    ps.push("a", a->dv);
    ps.push("b", b->dv);
    std::cout << "param:  "
              << "a -> " << ps.pull("a") << ","
              << "b -> " << ps.pull("b") << "\n";
  }
  return 0;
}