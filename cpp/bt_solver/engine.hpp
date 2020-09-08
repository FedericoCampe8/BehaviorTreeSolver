//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the solver engine.
//

#pragma once

#include <memory>  // for std::shared_ptr
#include <string>

#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS Engine {
 public:
  using SPtr = std::shared_ptr<Engine>;

 public:
  Engine();
  virtual ~Engine() = default;

  /// Registers the given instance/environment the evolution should run on.
  /// @note this method does not run on the environment.
  /// @throw std::runtime_error if this model is called while the engine is running
  //virtual void registerInstance(Engine::UPtr instance) = 0;

  /// Notifies the optimizer on a given EngineEvent
  //virtual void notifyEngine(const engine::Event& event) = 0;

  /// Wait for the current jobs in the queue to finish.
  /// @note if "timeoutSec" is -1, waits for all jobs in the queue to complete
  virtual void engineWait(int timeoutMsec) = 0;

  /// Shuts down the engine
  virtual void turnDown() = 0;
};

}  // namespace btsolver
