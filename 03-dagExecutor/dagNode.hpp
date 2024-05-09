#pragma once
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
namespace dag {
// 0 is invalid for id;
template <typename T>
struct IdAllocator {
  static T alloc() {
    static T nextId{1};
    return nextId++;
  }
};
using DataNodeId = uint32_t;
struct DataNodeIdAllocator {
  static DataNodeId alloc() {
    static DataNodeId nextId{1};
    return nextId++;
  }
};
using ProcessNodeId = uint32_t;
struct ProcessNodeIdAllocator {
  static ProcessNodeId alloc() {
    static ProcessNodeId nextId{1};
    return nextId++;
  }
};
using ProcessNodeTypeId = uint32_t;
// 0 is invalid for process node type id;
struct ProcessNodeTypeIdManager {
  template <typename T>
  static ProcessNodeTypeId get() {
    static ProcessNodeTypeId id = nextTypeId_++;
    return id;
  }

 private:
  static ProcessNodeTypeId nextTypeId_;
};
ProcessNodeTypeId ProcessNodeTypeIdManager::nextTypeId_{1};

struct ProcessNode {
  ProcessNodeTypeId processNodeType_{0};
  std::string processNodeName_;
  std::vector<DataNodeId> inputDataNodes_;
  std::vector<DataNodeId> outputDataNodes_;
};
template <typename T>
std::string toString(const std::vector<T>& vs) {
  std::stringstream ss;
  ss << "[";
  for (auto& v : vs) {
    ss << v << ",";
  }
  ss << "]";
  return ss.str();
}
std::string toString(const ProcessNode& node) {
  std::stringstream ss;
  ss << node.processNodeType_ << " " << node.processNodeName_ << " "
     << toString(node.inputDataNodes_) << " -> "
     << toString(node.outputDataNodes_);
  ;
  return ss.str();
}
}