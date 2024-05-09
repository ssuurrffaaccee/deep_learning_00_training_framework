#pragma once
#include <string>
#include <vector>
#include "dagNode.hpp"
namespace model{
  struct Model;
}
namespace function {
#define VIRTUAL_DESTRUCTOR(NAME) \
  virtual ~NAME() {}
struct ProcessNodeFunc {
  ProcessNodeFunc() {}
  ProcessNodeFunc(bool isBackwardFunc) : isBackwardFunc_{isBackwardFunc} {}
  VIRTUAL_DESTRUCTOR(ProcessNodeFunc)
  virtual void operator()(model::Model &model,
                          const std::vector<dag::DataNodeId> &inputs,
                          const std::vector<dag::DataNodeId> &outputs) = 0;
  virtual const std::string &typeName() = 0;
  bool isBackwardFunc_{false};
};
}  // namespace function