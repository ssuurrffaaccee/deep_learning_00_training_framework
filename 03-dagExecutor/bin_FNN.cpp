#include <iostream>

#include "functionImpl.hpp"
#include "model.hpp"
#include "modelExecutor.hpp"
#include "op.hpp"
int main() {
  try {
    bool isNeedBackward{true};
    model::Model model{isNeedBackward};
    auto A = ops::parameterf(model);
    auto x = ops::constf(model);
    auto b = ops::parameterf(model);
    auto y = ops::binary<function::Add>(ops::binary<function::Mul>(A, x, model),b,model);
    auto o = ops::map<function::Relu>(y, model);
    auto r = ops::constf(model);
    auto l = ops::binary<function::LossL2>(o,r,model);
    ops::backward(l,model);
    ModelExecutor me;
    auto result = me.execute(model);
    result.dump("bin_FNN.dot");
  } catch (MyExceptoin &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}