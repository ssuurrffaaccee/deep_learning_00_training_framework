#include <iostream>

#include "functionImpl.hpp"
#include "model.hpp"
#include "modelExecutor.hpp"
#include "op.hpp"
model::DataNode vectorSum(const std::vector<model::DataNode> &v1,
                          model::Model &model) {
  CHECK(!v1.empty());
  auto sum = v1[0];
  for (int i = 1; i < v1.size(); i++) {
    sum = ops::binary<function::Add>(sum, v1[i], model);
  }
  return sum;
}
model::DataNode vectorLoss(const std::vector<model::DataNode> &v1,
                           const std::vector<model::DataNode> &v2,
                           model::Model &model) {
  CHECK(!v1.empty());
  CHECK(!v2.empty());
  CHECK(v1.size() == v2.size());
  std::vector<model::DataNode> res;
  for (int i = 0; i < v1.size(); i++) {
    res.push_back(ops::binary<function::LossL2>(v2[i], v1[i], model));
  }
  return vectorSum(res, model);
}
int main() {
  try {
    bool isNeedBackward{true};
    model::Model model{isNeedBackward};
    auto inA = ops::parameterf(model);
    auto inB = ops::parameterf(model);
    auto inC = ops::parameterf(model);
    auto outA = ops::parameterf(model);
    auto outB = ops::parameterf(model);
    int step = 3;
    auto initState = ops::constf(model);

    std::vector<model::DataNode> inputSeries;
    for (int i = 0; i < step; i++) {
      inputSeries.push_back(ops::constf(model));
    }
    auto state = initState;
    std::vector<model::DataNode> outputSeries;
    for (int i = 0; i < step; i++) {
      auto internalState = ops::binary<function::Mul>(inA, state, model);
      auto internalInput =
          ops::binary<function::Mul>(inB, inputSeries[i], model);
      auto internalOutput = ops::binary<function::Add>(
          ops::binary<function::Add>(internalState, internalInput, model), inC,
          model);
      auto activateInternalOutput =
          ops::map<function::Relu>(internalOutput, model);
      state = ops::binary<function::Mul>(outA, internalState, model);
      auto output =
          ops::binary<function::Mul>(outB, activateInternalOutput, model);
      outputSeries.push_back(output);
    }
    std::vector<model::DataNode> groundTruthSeries;
    for (int i = 0; i < step; i++) {
      groundTruthSeries.push_back(ops::constf(model));
    }
    auto loss = vectorLoss(outputSeries, groundTruthSeries, model);
    ops::backward(loss, model);
    ModelExecutor me;
    auto result = me.execute(model);
    result.dump("bin_Autoregressive.dot");
  } catch (MyExceptoin &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}