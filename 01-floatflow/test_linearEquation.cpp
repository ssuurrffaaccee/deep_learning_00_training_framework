#include "floatflow.h"
#include <iostream>
#include <random>
int main() {
  ParameterServer ps = ParameterServer();
  ps.regist(ParameterServer::Value{
      "x", "adam", {{"lr", 0.1}, {"beta0", 0.9}, {"beta1", 0.99}}, -1});
  // 0.3*x + 0.4 = 0.8 ===> x = 4/3
  Tape t = Tape();
  DValue *a = constant(0.3);
  DValue *b = constant(0.4);
  DValue *x = parameter();
  DValue *z = linear(a, x, b, &t);
  DValue *y = constant(0.8);
  DValue *loss = L2_(z, y, &t);

  for (int i = 0; i < 10000; i++) {
    x->v = ps.pull("x");
    t.forward();
    loss->dv = 1.0f;
    t.backward();
    ps.push("x", x->dv);
    std::cout << "x: " << ps.pull("x") << "\n";
  }
  return 0;
}