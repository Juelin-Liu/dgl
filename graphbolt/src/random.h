
/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file random.h
 * @brief Random Engine class.
 */
#ifndef GRAPHBOLT_RANDOM_H_
#define GRAPHBOLT_RANDOM_H_

#include <dmlc/thread_local.h>

#include <optional>
#include <pcg_random.hpp>
#include <random>
#include <thread>

namespace graphbolt {

/**
 * @brief Thread-local Random Number Generator class.
 */
class RandomEngine {
 public:
  /** @brief Constructor with default seed. */
  RandomEngine();

  /** @brief Constructor with given seed. */
  explicit RandomEngine(uint64_t seed);
  explicit RandomEngine(uint64_t seed, uint64_t stream);

  /** @brief Get the thread-local random number generator instance. */
  static RandomEngine* ThreadLocal();

  /** @brief Set the seed. */
  void SetSeed(uint64_t seed);
  void SetSeed(uint64_t seed, uint64_t stream);

  /** @brief Manually fix the seed. */
  static std::optional<uint64_t> manual_seed;
  static void SetManualSeed(int64_t seed);

  /**
   * @brief Generate a uniform random integer in [low, high).
   */
  template <typename T>
  T RandInt(T lower, T upper) {
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

  /**
   * @brief Generate a uniform random real number in [low, high).
   */
  template <typename T>
  T Uniform(T lower, T upper) {
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }

  /**
   * @brief Generate random non-negative floating-point values according to
   * exponential distribution. Probability density function: P(x|λ) = λe^(-λx).
   */
  template <typename T>
  T Exponential(T lambda) {
    std::exponential_distribution<T> dist(lambda);
    return dist(rng_);
  }

 private:
  pcg32 rng_;
};

namespace labor {

template <typename T>
inline T uniform_random(int64_t random_seed, int64_t t) {
  pcg32 ng(random_seed, t);
  std::uniform_real_distribution<T> uni;
  return uni(ng);
}

template <typename T>
inline T invcdf(T u, int64_t n, T rem) {
  constexpr T one = 1;
  return rem * (one - std::pow(one - u, one / n));
}

template <typename T>
inline T jth_sorted_uniform_random(
    int64_t random_seed, int64_t t, int64_t c, int64_t j, T& rem, int64_t n) {
  const auto u = uniform_random<T>(random_seed, t + j * c);
  // https://mathematica.stackexchange.com/a/256707
  rem -= invcdf(u, n, rem);
  return 1 - rem;
}

};  // namespace labor

}  // namespace graphbolt

#endif  // GRAPHBOLT_RANDOM_H_
