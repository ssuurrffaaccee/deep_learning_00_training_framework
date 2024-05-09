#pragma once
#include <iostream>

#include "dagExecutor.hpp"
namespace model {
enum class DataNodeUsage {
  UNKOWN,
  TEMP,
  PARAMETER,
  CONST,
};
struct DataNode {
  DataNode() {}
  dag::DataNodeId id_{0};
  bool isNeedGrad_{false};
  DataNodeUsage usage_{DataNodeUsage::UNKOWN};
};
DataNode buildDataNode(bool isNeedGrad, const DataNodeUsage &usage) {
  DataNode newDataNode;
  newDataNode.id_ = dag::DataNodeIdAllocator::alloc();
  newDataNode.isNeedGrad_ = isNeedGrad;
  newDataNode.usage_ = usage;
  return newDataNode;
}
DataNode buildDataNode(dag::DataNodeId id, bool isNeedGrad,
                       const DataNodeUsage &usage) {
  DataNode newDataNode;
  newDataNode.id_ = id;
  newDataNode.isNeedGrad_ = isNeedGrad;
  newDataNode.usage_ = usage;
  return newDataNode;
}
DataNode fakeDataNode() {
  DataNode newDataNode;
  newDataNode.id_ = 0;
  newDataNode.isNeedGrad_ = false;
  newDataNode.usage_ = DataNodeUsage::TEMP;
  return newDataNode;
}
DataNode buildDataNodeByInference(const DataNode &n0, const DataNode &n1) {
  DataNode newDataNode;
  newDataNode.id_ = dag::DataNodeIdAllocator::alloc();
  if (n0.isNeedGrad_ || n1.isNeedGrad_) {
    newDataNode.isNeedGrad_ = true;
  } else {
    newDataNode.isNeedGrad_ = false;
  }
  newDataNode.usage_ = DataNodeUsage::TEMP;
  return newDataNode;
}
DataNode buildDataNodeByInference(const DataNode &n0, const DataNode &n1, const DataNode &n2) {
  DataNode newDataNode;
  newDataNode.id_ = dag::DataNodeIdAllocator::alloc();
  if (n0.isNeedGrad_ || n1.isNeedGrad_ || n2.isNeedGrad_) {
    newDataNode.isNeedGrad_ = true;
  } else {
    newDataNode.isNeedGrad_ = false;
  }
  newDataNode.usage_ = DataNodeUsage::TEMP;
  return newDataNode;
}
DataNode buildDataNodeByInference(const DataNode &n0) {
  DataNode newDataNode;
  newDataNode.id_ = dag::DataNodeIdAllocator::alloc();
  if (n0.isNeedGrad_) {
    newDataNode.isNeedGrad_ = true;
  } else {
    newDataNode.isNeedGrad_ = false;
  }
  newDataNode.usage_ = DataNodeUsage::TEMP;
  return newDataNode;
}

struct ProcessNode {
  dag::ProcessNodeTypeId processNodeType_{0};
  std::string processNodeName_;
  std::vector<DataNode> inputDataNodes_;
  std::vector<DataNode> outputDataNodes_;
};
dag::ProcessNode toDagProcessNode(const ProcessNode &processNode) {
  dag::ProcessNode dagProcessNode;
  dagProcessNode.processNodeType_ = processNode.processNodeType_;
  dagProcessNode.processNodeName_ = processNode.processNodeName_;
  for (auto &inputNode : processNode.inputDataNodes_) {
    dagProcessNode.inputDataNodes_.push_back(inputNode.id_);
  }
  for (auto &outputNode : processNode.outputDataNodes_) {
    dagProcessNode.outputDataNodes_.push_back(outputNode.id_);
  }
  return dagProcessNode;
}
struct Model {
  Model(bool isNeedBackward): isNeedBackward_{isNeedBackward} {}
  void addForward(const ProcessNode &node) { forwardNodes_.push_back(node); }
  void addBackWard(const ProcessNode &node) { backwardNodes_.push_back(node); }
  void recordGradDataNodeID(const dag::DataNodeId &data,
                            const dag::DataNodeId &grad) {
    CHECK(dataToGradMap_.find(data) == dataToGradMap_.end());
    dataToGradMap_[data] = grad;
  }
  dag::DataNodeId getGradNode(const dag::DataNodeId &data) {
    auto iter = dataToGradMap_.find(data);
    CHECK_WITH_INFO(iter != dataToGradMap_.end(), std::to_string(data));
    return iter->second;
  }
  const std::vector<ProcessNode> &forwardNodes() const & {
    return forwardNodes_;
  }
  const std::vector<ProcessNode> &backwardNodes() const & {
    return backwardNodes_;
  }
  void addInput(const std::string &name, const DataNode &id) {
    CHECK(inputs_.find(name) == inputs_.end());
    inputs_[name] = id;
  }
  void addOutput(const std::string &name, const DataNode &id) {
    CHECK(outputs_.find(name) == outputs_.end());
    outputs_[name] = id;
  }
  void addFuncs(const dag::ProcessNodeTypeId &type,
                const std::string &procesNodeName,
                std::shared_ptr<function::ProcessNodeFunc> func) {
    processFuncs_[type][procesNodeName] = func;
  }
  std::shared_ptr<function::ProcessNodeFunc>
  getFunc(dag::ProcessNodeTypeId type, const std::string &procesNodeName) {
    auto iterTypeFuncs = processFuncs_.find(type);
    CHECK(iterTypeFuncs != processFuncs_.end());
    auto iterFunc = iterTypeFuncs->second.find(procesNodeName);
    CHECK(iterFunc != iterTypeFuncs->second.end());
    return iterFunc->second;
  }
  bool isNeedBackward_{false};
private:
  std::unordered_map<std::string, DataNode> inputs_;
  std::unordered_map<std::string, DataNode> outputs_;
  std::unordered_map<dag::DataNodeId, dag::DataNodeId> dataToGradMap_;
  std::vector<ProcessNode> forwardNodes_;
  std::vector<ProcessNode> backwardNodes_;
  using ProcessNodeFuncMap = std::unordered_map<
      dag::ProcessNodeTypeId,
      std::unordered_map<std::string,
                         std::shared_ptr<function::ProcessNodeFunc>>>;
  ProcessNodeFuncMap processFuncs_;
};
}; // namespace model