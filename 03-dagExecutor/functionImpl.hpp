#pragma once
#include "function.hpp"
namespace function {

template <typename T>
struct GetGradType;

struct Add : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(Add)
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"add"};
    return name_;
  }
};
struct AddGrad : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(AddGrad)
  AddGrad() : ProcessNodeFunc{true} {};
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"add_grad"};
    return name_;
  }
};
template <>
struct GetGradType<Add> {
  using type = AddGrad;
};
struct Mul : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(Mul)
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"mul"};
    return name_;
  }
};
struct MulGrad : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(MulGrad)
  MulGrad() : ProcessNodeFunc{true} {};
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"mul_grad"};
    return name_;
  }
};
template <>
struct GetGradType<Mul> {
  using type = MulGrad;
};
struct Backward : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(Backward)
  Backward() : ProcessNodeFunc{true} {}
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"backward"};
    return name_;
  }
};
struct Const : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(Const)
  Const() {}
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"const"};
    return name_;
  }
};

#define FUNCTION(NAME, FORWARD_STR_NAME, BACKWARD_STRANME)                   \
  struct NAME : public ProcessNodeFunc {                                     \
    VIRTUAL_DESTRUCTOR(NAME)                                                 \
    NAME() {}                                                                \
    void operator()(model::Model &model,                                     \
                    const std::vector<dag::DataNodeId> &inputs,              \
                    const std::vector<dag::DataNodeId> &outputs) override {} \
    const std::string &typeName() override {                                 \
      static std::string name_{FORWARD_STR_NAME};                            \
      return name_;                                                          \
    }                                                                        \
  };                                                                         \
  struct NAME##Grad : public ProcessNodeFunc {                               \
    VIRTUAL_DESTRUCTOR(NAME##Grad)                                           \
    NAME##Grad() : ProcessNodeFunc{true} {}                                  \
    void operator()(model::Model &model,                                     \
                    const std::vector<dag::DataNodeId> &inputs,              \
                    const std::vector<dag::DataNodeId> &outputs) override {} \
    const std::string &typeName() override {                                 \
      static std::string name_{BACKWARD_STRANME};                            \
      return name_;                                                          \
    }                                                                        \
  };                                                                         \
  template <>                                                                \
  struct GetGradType<NAME> {                                                 \
    using type = NAME##Grad;                                                 \
  };

FUNCTION(Relu, "relu", "relu_grad")
FUNCTION(Parameter, "parameter", "parameter_grad")
FUNCTION(LossL2, "lossl2", "lossl2_grad")
FUNCTION(AutoRegressive, "autoregressive", "autoregressive_grad")
struct Input : public ProcessNodeFunc {
  VIRTUAL_DESTRUCTOR(Input)
  Input() {}
  void operator()(model::Model &model,
                  const std::vector<dag::DataNodeId> &inputs,
                  const std::vector<dag::DataNodeId> &outputs) override {}
  const std::string &typeName() override {
    static std::string name_{"input"};
    return name_;
  }
};
}  // namespace function