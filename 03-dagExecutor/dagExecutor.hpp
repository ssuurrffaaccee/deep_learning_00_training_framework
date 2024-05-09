#pragma once
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "check.hpp"
#include "dagNode.hpp"
namespace dag {
struct DagExecutor {
  using ExecutteFunction = std::function<void(
      const ProcessNodeId &, const std::string &,
      const std::vector<DataNodeId> &, const std::vector<DataNodeId> &)>;
  std::optional<std::vector<ProcessNode>> nextParallelProcessNodeGroup() {
    if (frontProcessNodes_.empty()) {
      return std::optional<std::vector<ProcessNode>>{};
    }
    std::set<std::string> newFrontProcessNodes;
    std::cout << "Potential parallel num:" << frontProcessNodes_.size() << "\n";
    for (auto &processNodeName : frontProcessNodes_) {
      auto &processNode = processNodes_[processNodeName];
      for (auto &ouputDataNodeId : processNode.outputDataNodes_) {
        auto iter = dataNodeToConsumeProcesNodes_.find(ouputDataNodeId);
        if (iter == dataNodeToConsumeProcesNodes_.end()) {
          continue;
        }
        auto &outputProcessNodes = iter->second;
        std::set<std::string> noNeedRecordProcessNodes;
        for (auto &maybeNeedActiveProcessNodeName : outputProcessNodes) {
          if (maybeNeedActiveProcessNodeName == "12_autoregressive_grad") {
            std::cout << maybeNeedActiveProcessNodeName << " "
                      << ouputDataNodeId << "\n";
          }
          uint32_t &inArcNum =
              processToInArcNumMap_[maybeNeedActiveProcessNodeName];
          inArcNum--;
          if (inArcNum == 0) {
            newFrontProcessNodes.insert(maybeNeedActiveProcessNodeName);
            noNeedRecordProcessNodes.insert(maybeNeedActiveProcessNodeName);
          }
        }
        for (auto &noNeedRecordProcessNode : noNeedRecordProcessNodes) {
          outputProcessNodes.erase(noNeedRecordProcessNode);
        }
      }
    }
    std::swap(frontProcessNodes_, newFrontProcessNodes);
    std::vector<ProcessNode> executableProcessNodes;
    for (auto &processNodeName : newFrontProcessNodes) {
      auto &processNode = processNodes_[processNodeName];
      executableProcessNodes.push_back(processNode);
    }
    return std::optional<std::vector<ProcessNode>>(
        std::move(executableProcessNodes));
  }
  void add(const ProcessNode &processNode) {
    std::cout << toString(processNode) << "\n";
    auto &processNodeName = processNode.processNodeName_;
    // check process node not repeate
    CHECK_WITH_INFO(processNodes_.find(processNodeName) == processNodes_.end(),
                    processNode.processNodeName_);
    processNodes_[processNodeName] = processNode;
    uint32_t inArcNum = processNode.inputDataNodes_.size();
    if (inArcNum == 0) {
      frontProcessNodes_.insert(processNodeName);
    } else {
      processToInArcNumMap_[processNodeName] = inArcNum;
    }
    for (auto &dataNodeId : processNode.inputDataNodes_) {
      dataNodeToConsumeProcesNodes_[dataNodeId].insert(processNodeName);
    }
  }

 private:
  // topological aux info
  std::unordered_map<std::string, uint32_t> processToInArcNumMap_;
  std::unordered_map<DataNodeId, std::multiset<std::string>>
      dataNodeToConsumeProcesNodes_;
  std::unordered_map<std::string, ProcessNode> processNodes_;
  std::set<std::string> frontProcessNodes_;
};
}  // namespace dag