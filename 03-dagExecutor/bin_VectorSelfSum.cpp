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
std::vector<model::DataNode> vectorSum(const std::vector<model::DataNode> &v1,
                                       const std::vector<model::DataNode> &v2,
                                       model::Model &model) {
  CHECK(!v1.empty());
  CHECK(!v2.empty());
  CHECK(v1.size() == v2.size());
  std::vector<model::DataNode> res;
  for (int i = 0; i < v1.size(); i++) {
    res.push_back(ops::binary<function::Add>(v2[i], v1[i], model));
  }
  return res;
}

model::DataNode vectorMulSum(const std::vector<model::DataNode> &v1,
                             const std::vector<model::DataNode> &v2,
                             model::Model &model) {
  CHECK(v1.size() == v2.size());
  std::vector<model::DataNode> mulRes;
  for (int i = 0; i < v1.size(); i++) {
    mulRes.push_back(ops::binary<function::Mul>(v1[i], v2[i], model));
  }
  return vectorSum(mulRes, model);
}

std::vector<model::DataNode> matrixVectorMul(
    const std::vector<std::vector<model::DataNode>> &m,
    const std::vector<model::DataNode> &v, model::Model &model) {
  std::vector<model::DataNode> res;
  for (auto &mv : m) {
    res.push_back(vectorMulSum(mv, v, model));
  }
  return res;
}
int main() {
  try {
    bool isNeedBackward{true};
    model::Model model{isNeedBackward};
    std::vector<model::DataNode> v{{ops::parameterf(model),
                                    ops::parameterf(model),
                                    ops::parameterf(model)}};
    auto r = vectorSum(v,v,model);
    auto f = vectorSum(r,model);
    ops::backward(f, model);
    ModelExecutor me;
    auto result = me.execute(model);
    result.dump("bin_VectorSelfSum.dot");
  } catch (MyExceptoin &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}