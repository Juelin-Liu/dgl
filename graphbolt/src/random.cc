/**
 *  Copyright (c) 2023 by Contributors
 * @file random.cc
 * @brief Random Engine.
 */

#include "./random.h"

#include <torch/torch.h>

namespace graphbolt {

namespace {

// Get a unique integer ID representing this thread.
inline uint32_t GetThreadId() {
  static int num_threads = 0;
  static std::mutex mutex;
  static thread_local int id = -1;

  if (id == -1) {
    std::lock_guard<std::mutex> guard(mutex);
    id = num_threads;
    num_threads++;
  }
  return id;
}

};  // namespace

std::optional<uint64_t> RandomEngine::manual_seed;

/** @brief Constructor with default seed. */
RandomEngine::RandomEngine() {
  std::random_device rd;
  uint64_t seed = manual_seed.value_or(rd());
  SetSeed(seed);
}

/** @brief Constructor with given seed. */
RandomEngine::RandomEngine(uint64_t seed) { RandomEngine(seed, GetThreadId()); }

/** @brief Constructor with given seed. */
RandomEngine::RandomEngine(uint64_t seed, uint64_t stream) {
  SetSeed(seed, stream);
}

/** @brief Get the thread-local random number generator instance. */
RandomEngine* RandomEngine::ThreadLocal() {
  return dmlc::ThreadLocalStore<RandomEngine>::Get();
}

/** @brief Set the seed. */
void RandomEngine::SetSeed(uint64_t seed) { SetSeed(seed, GetThreadId()); }

/** @brief Set the seed. */
void RandomEngine::SetSeed(uint64_t seed, uint64_t stream) {
  rng_.seed(seed, stream);
}

/** @brief Manually fix the seed. */
void RandomEngine::SetManualSeed(int64_t seed) { manual_seed = seed; }

}  // namespace graphbolt
