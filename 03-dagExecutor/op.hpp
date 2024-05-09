#pragma once
#include "function.hpp"
#include "model.hpp"
namespace ops {
model::DataNode input(const std::string &name, model::Model &m) {
  auto dataNode = model::buildDataNode(false, model::DataNodeUsage::TEMP);
  m.addInput(name, dataNode);
  return dataNode;
}
model::DataNode constf(model::Model &m) {
  using T = function::Const;
  auto func = std::make_shared<T>();
  dag::ProcessNodeTypeId type = dag::ProcessNodeTypeIdManager::get<T>();
  dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
  std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
  m.addFuncs(type, procesNodeName,
             std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
  model::ProcessNode newProcessNode;
  newProcessNode.processNodeType_ = type;
  newProcessNode.processNodeName_ = procesNodeName;
  auto z = model::buildDataNode(false, model::DataNodeUsage::CONST);
  newProcessNode.outputDataNodes_.push_back(z);
  m.addForward(newProcessNode);
  return z;
}
model::DataNode parameterf(model::Model &m) {
  using T = function::Parameter;
  auto func = std::make_shared<T>();
  dag::ProcessNodeTypeId type = dag::ProcessNodeTypeIdManager::get<T>();
  dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
  std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
  m.addFuncs(type, procesNodeName,
             std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
  model::ProcessNode newProcessNode;
  newProcessNode.processNodeType_ = type;
  newProcessNode.processNodeName_ = procesNodeName;
  auto x =
      model::buildDataNode(m.isNeedBackward_, model::DataNodeUsage::PARAMETER);
  newProcessNode.outputDataNodes_.push_back(x);
  m.addForward(newProcessNode);
  if (m.isNeedBackward_) {
    auto dx = model::buildDataNode(false, model::DataNodeUsage::TEMP);
    m.recordGradDataNodeID(x.id_, dx.id_);
    using GradFunc = typename function::GetGradType<T>::type;
    auto func = std::make_shared<GradFunc>();
    dag::ProcessNodeTypeId type =
        dag::ProcessNodeTypeIdManager::get<GradFunc>();
    dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
    std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
    m.addFuncs(type, procesNodeName,
               std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
    model::ProcessNode newProcessNode;
    newProcessNode.processNodeType_ = type;
    newProcessNode.processNodeName_ = procesNodeName;
    newProcessNode.outputDataNodes_.push_back(dx);
    m.addBackWard(newProcessNode);
  }
  return x;
}
template <typename T>
model::DataNode binary(const model::DataNode &x, const model::DataNode &y,
                       model::Model &m) {
  auto func = std::make_shared<T>();
  dag::ProcessNodeTypeId type = dag::ProcessNodeTypeIdManager::get<T>();
  dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
  std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
  m.addFuncs(type, procesNodeName,
             std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
  model::ProcessNode newProcessNode;
  newProcessNode.processNodeType_ = type;
  newProcessNode.processNodeName_ = procesNodeName;
  newProcessNode.inputDataNodes_.push_back(x);
  newProcessNode.inputDataNodes_.push_back(y);
  auto z = model::buildDataNodeByInference(x, y);
  newProcessNode.outputDataNodes_.push_back(z);
  m.addForward(newProcessNode);
  if (x.isNeedGrad_ || y.isNeedGrad_) {
    model::DataNode dx = model::fakeDataNode();
    model::DataNode dy = model::fakeDataNode();
    if (x.isNeedGrad_) {
      dx = model::buildDataNode(m.getGradNode(x.id_), false,
                                model::DataNodeUsage::TEMP);
    }
    if (y.isNeedGrad_) {
      dy = model::buildDataNode(m.getGradNode(y.id_), false,
                                model::DataNodeUsage::TEMP);
    }
    auto dz = model::buildDataNode(false, model::DataNodeUsage::TEMP);
    m.recordGradDataNodeID(z.id_, dz.id_);
    using GradFunc = typename function::GetGradType<T>::type;
    auto func = std::make_shared<GradFunc>();
    dag::ProcessNodeTypeId type =
        dag::ProcessNodeTypeIdManager::get<GradFunc>();
    dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
    std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
    m.addFuncs(type, procesNodeName,
               std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
    model::ProcessNode newProcessNode;
    newProcessNode.processNodeType_ = type;
    newProcessNode.processNodeName_ = procesNodeName;
    newProcessNode.inputDataNodes_.push_back(x);
    newProcessNode.inputDataNodes_.push_back(y);
    newProcessNode.inputDataNodes_.push_back(z);
    newProcessNode.inputDataNodes_.push_back(dz);
    newProcessNode.outputDataNodes_.push_back(dx);
    newProcessNode.outputDataNodes_.push_back(dy);
    m.addBackWard(newProcessNode);
  }
  return z;
}
template <typename T>
std::pair<model::DataNode, model::DataNode> in2P1ToOut2(
    const model::DataNode &x, const model::DataNode &y,
    const model::DataNode &parameter, model::Model &m) {
  auto func = std::make_shared<T>();
  dag::ProcessNodeTypeId type = dag::ProcessNodeTypeIdManager::get<T>();
  dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
  std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
  m.addFuncs(type, procesNodeName,
             std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
  model::ProcessNode newProcessNode;
  newProcessNode.processNodeType_ = type;
  newProcessNode.processNodeName_ = procesNodeName;
  newProcessNode.inputDataNodes_.push_back(x);
  newProcessNode.inputDataNodes_.push_back(y);
  newProcessNode.inputDataNodes_.push_back(parameter);
  auto z0 = model::buildDataNodeByInference(x, y, parameter);
  newProcessNode.outputDataNodes_.push_back(z0);
  auto z1 = model::buildDataNodeByInference(x, y, parameter);
  newProcessNode.outputDataNodes_.push_back(z1);
  m.addForward(newProcessNode);
  if (x.isNeedGrad_ || y.isNeedGrad_ || parameter.isNeedGrad_) {
    model::DataNode dx = model::fakeDataNode();
    model::DataNode dy = model::fakeDataNode();
    model::DataNode dparameter = model::fakeDataNode();
    if (x.isNeedGrad_) {
      dx = model::buildDataNode(m.getGradNode(x.id_), false,
                                model::DataNodeUsage::TEMP);
    }
    if (y.isNeedGrad_) {
      dy = model::buildDataNode(m.getGradNode(y.id_), false,
                                model::DataNodeUsage::TEMP);
    }
    if (parameter.isNeedGrad_) {
      dparameter = model::buildDataNode(m.getGradNode(parameter.id_), false,
                                        model::DataNodeUsage::TEMP);
    }
    auto dz0 = model::buildDataNode(false, model::DataNodeUsage::TEMP);
    m.recordGradDataNodeID(z0.id_, dz0.id_);
    auto dz1 = model::buildDataNode(false, model::DataNodeUsage::TEMP);
    m.recordGradDataNodeID(z1.id_, dz1.id_);
    using GradFunc = typename function::GetGradType<T>::type;
    auto func = std::make_shared<GradFunc>();
    dag::ProcessNodeTypeId type =
        dag::ProcessNodeTypeIdManager::get<GradFunc>();
    dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
    std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
    m.addFuncs(type, procesNodeName,
               std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
    model::ProcessNode newProcessNode;
    newProcessNode.processNodeType_ = type;
    newProcessNode.processNodeName_ = procesNodeName;
    newProcessNode.inputDataNodes_.push_back(x);
    newProcessNode.inputDataNodes_.push_back(y);
    newProcessNode.inputDataNodes_.push_back(parameter);
    newProcessNode.inputDataNodes_.push_back(z0);
    newProcessNode.inputDataNodes_.push_back(dz0);
    newProcessNode.inputDataNodes_.push_back(z1);
    newProcessNode.inputDataNodes_.push_back(dz1);
    newProcessNode.outputDataNodes_.push_back(dx);
    newProcessNode.outputDataNodes_.push_back(dy);
    newProcessNode.outputDataNodes_.push_back(dparameter);
    m.addBackWard(newProcessNode);
  }
  return std::make_pair(z0, z1);
}
template <typename T>
model::DataNode map(const model::DataNode &x, model::Model &m) {
  auto func = std::make_shared<T>();
  dag::ProcessNodeTypeId type = dag::ProcessNodeTypeIdManager::get<T>();
  dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
  std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
  m.addFuncs(type, procesNodeName,
             std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
  model::ProcessNode newProcessNode;
  newProcessNode.processNodeType_ = type;
  newProcessNode.processNodeName_ = procesNodeName;
  newProcessNode.inputDataNodes_.push_back(x);
  auto z = model::buildDataNodeByInference(x);
  newProcessNode.outputDataNodes_.push_back(z);
  m.addForward(newProcessNode);
  if (x.isNeedGrad_) {
    model::DataNode dx = model::buildDataNode(m.getGradNode(x.id_), false,
                                              model::DataNodeUsage::TEMP);
    auto dz = model::buildDataNode(false, model::DataNodeUsage::TEMP);
    m.recordGradDataNodeID(z.id_, dz.id_);
    using GradFunc = typename function::GetGradType<T>::type;
    auto func = std::make_shared<GradFunc>();
    dag::ProcessNodeTypeId type =
        dag::ProcessNodeTypeIdManager::get<GradFunc>();
    dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
    std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
    m.addFuncs(type, procesNodeName,
               std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
    model::ProcessNode newProcessNode;
    newProcessNode.processNodeType_ = type;
    newProcessNode.processNodeName_ = procesNodeName;
    newProcessNode.inputDataNodes_.push_back(x);
    newProcessNode.inputDataNodes_.push_back(z);
    newProcessNode.inputDataNodes_.push_back(dz);
    newProcessNode.outputDataNodes_.push_back(dx);
    m.addBackWard(newProcessNode);
  }
  return z;
}
void backward(model::DataNode &x, model::Model &m) {
  if (x.isNeedGrad_) {
    auto func = std::make_shared<function::Backward>();
    dag::ProcessNodeTypeId type =
        dag::ProcessNodeTypeIdManager::get<function::Backward>();
    dag::ProcessNodeId id = dag::ProcessNodeIdAllocator::alloc();
    std::string procesNodeName = std::to_string(id) + "_" + func->typeName();
    m.addFuncs(type, procesNodeName,
               std::dynamic_pointer_cast<function::ProcessNodeFunc>(func));
    model::ProcessNode newProcessNode;
    newProcessNode.processNodeType_ = type;
    newProcessNode.processNodeName_ = procesNodeName;
    auto dx = model::buildDataNode(m.getGradNode(x.id_), false,
                                   model::DataNodeUsage::TEMP);
    newProcessNode.outputDataNodes_.push_back(dx);
    m.addForward(newProcessNode);
  }
}
}  // namespace ops