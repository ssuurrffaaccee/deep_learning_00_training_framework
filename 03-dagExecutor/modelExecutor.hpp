#pragma once
#include <fstream>

#include "dagExecutor.hpp"
#include "model.hpp"
struct ExecutionResult {
  void addParallelOps(
      std::vector<std::pair<dag::ProcessNode, bool>> &processNodes) {
    std::set<std::string> forwardParallelGroup;
    std::set<std::string> backwardParallelGroup;
    for (auto &[processNode, isBackward] : processNodes) {
      if (isBackward) {
        addOp(processNode, backwardParallelGroup);
      } else {
        addOp(processNode, forwardParallelGroup);
      }
    }
    if (!forwardParallelGroup.empty()) {
      forwardParallelGroups_.push_back(forwardParallelGroup);
    }
    if (!backwardParallelGroup.empty()) {
      backwardParallelGroups_.push_back(backwardParallelGroup);
    }
  }
  void addOp(const dag::ProcessNode &processNode,
             std::set<std::string> &opCluster_) {
    auto processNodeStr = "p_" + processNode.processNodeName_;
    opCluster_.insert(processNodeStr);
    graphVizEdgeLines_.push_back(processNodeStr + " [shape=ellipse]");
    for (auto &inputNode : processNode.inputDataNodes_) {
      auto inputStr = "d_" + std::to_string(inputNode);
      opCluster_.insert(inputStr);
      graphVizEdgeLines_.push_back(inputStr + " -> " + processNodeStr);
    }
    for (auto &outputNode : processNode.outputDataNodes_) {
      if (outputNode == 0) {
        continue;
      }
      auto outputStr = "d_" + std::to_string(outputNode);
      opCluster_.insert(outputStr);
      graphVizEdgeLines_.push_back(processNodeStr + " -> " + outputStr);
    }
  }
  void dump(const std::string &path) {
    auto outFile = std::ofstream(path);
    CHECK(outFile.is_open());
    outFile << "digraph {\n";
    outFile << "    rankdir=LR\n";
    outFile << "    node [ shape=square ];\n";
    outFile << "  subgraph cluster_forward {\n";
    outFile << "        label=Forward;\n";
    for (int i = 0; i < forwardParallelGroups_.size(); i++) {
      outFile << "    subgraph cluster_parallel_forward_" << std::to_string(i)
              << " {\n";
      outFile << "        label=ParallelCluster_forward_" << std::to_string(i)
              << ";\n";
      for (auto &opName : forwardParallelGroups_[i]) {
        outFile << "        " << opName << "\n";
      }
      outFile << "  }\n";
    }
    outFile << "    }\n";
    outFile << "  subgraph cluster_backward {\n";
    outFile << "        label=Backward;\n";
    for (int i = 0; i < backwardParallelGroups_.size(); i++) {
      outFile << "    subgraph cluster_parallel_backward_" << std::to_string(i)
              << " {\n";
      outFile << "        label=ParallelCluster_backward_" << std::to_string(i)
              << ";\n";
      for (auto &opName : backwardParallelGroups_[i]) {
        outFile << "        " << opName << "\n";
      }
      outFile << "    }\n";
    }
    outFile << "  }\n";
    for (auto &line : graphVizEdgeLines_) {
      outFile << "    " << line << "\n";
    }
    outFile << "}\n";
  }
  std::vector<std::string> forwardOps_;
  std::vector<std::string> backwardOps_;
  std::vector<std::string> graphVizEdgeLines_;
  std::vector<std::set<std::string>> parallelGroups_;
  std::vector<std::set<std::string>> forwardParallelGroups_;
  std::vector<std::set<std::string>> backwardParallelGroups_;
};
struct ModelExecutor {
  ExecutionResult execute(model::Model &model) {
    dag::DagExecutor dagExecutor;
    auto &forwardProcessNodes = model.forwardNodes();
    for (int i = 0; i < forwardProcessNodes.size(); i++) {
      dagExecutor.add(model::toDagProcessNode(forwardProcessNodes[i]));
    }
    auto &backwardProcessNodes = model.backwardNodes();
    for (int i = backwardProcessNodes.size() - 1; i >= 0; i--) {
      dagExecutor.add(model::toDagProcessNode(backwardProcessNodes[i]));
    }
    // dagExecutor.init();
    ExecutionResult result;
    while (true) {
      auto maybeNextGroup = dagExecutor.nextParallelProcessNodeGroup();
      if (!maybeNextGroup.has_value()) {
        return result;
      }
      auto &nextGroup = maybeNextGroup.value();
      std::vector<std::pair<dag::ProcessNode, bool>> parallelNodes;
      for (auto &processNode : nextGroup) {
        auto func = model.getFunc(processNode.processNodeType_,
                                  processNode.processNodeName_);
        CHECK(func != nullptr);
        parallelNodes.push_back(
            std::make_pair(processNode, func->isBackwardFunc_));
      }
      result.addParallelOps(parallelNodes);
    }
  }
};