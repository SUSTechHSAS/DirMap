#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

// All inputs are validated contiguous NumPy buffers. No Python API is used,
// so two queries can run concurrently without the interpreter's GIL.
extern "C" int dirmap_dijkstra(
    int n, const int64_t* offsets, const int32_t* neighbours,
    const double* weights, int query_count, const int32_t* sources,
    int target_count, const int32_t* targets,
    double* distances, int32_t* parents, int worker_count) noexcept {
  try {
    std::vector<uint8_t> is_target(n, 0);
    int unique_targets = 0;
    for (int i = 0; i < target_count; ++i) {
      if (!is_target[targets[i]]) ++unique_targets;
      is_target[targets[i]] = 1;
    }
    std::atomic<int> next{0}, error{0};
    auto work = [&]() noexcept {
      try {
        for (int query; (query = next.fetch_add(1)) < query_count;) {
          double* distance = distances + static_cast<int64_t>(query) * n;
          int32_t* parent = parents + static_cast<int64_t>(query) * n;
          std::fill(distance, distance + n, std::numeric_limits<double>::infinity());
          std::fill(parent, parent + n, -9999);
          using Item = std::pair<double, int32_t>;
          std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
          const int32_t source = sources[query];
          distance[source] = 0;
          queue.emplace(0, source);
          int remaining = unique_targets;
          while (!queue.empty()) {
            const auto [value, node] = queue.top();
            queue.pop();
            if (value != distance[node]) continue;
            // A settled target has its final distance and predecessor chain.
            // Queries without an explicit target set compute the whole row.
            if (target_count && is_target[node] && --remaining == 0) break;
            for (int64_t edge = offsets[node]; edge < offsets[node + 1]; ++edge) {
              const auto to = neighbours[edge];
              const double candidate = value + weights[edge];
              if (candidate < distance[to]) {
                distance[to] = candidate;
                parent[to] = node;
                queue.emplace(candidate, to);
              }
            }
          }
        }
      } catch (...) { error.store(1); }
    };
    std::vector<std::thread> workers;
    // Failure to create a helper thread still permits a serial computation.
    try {
      for (int i = 1; i < std::min({worker_count, query_count, 2}); ++i)
        workers.emplace_back(work);
    } catch (...) {}
    work();
    for (auto& worker : workers) worker.join();
    return error.load() ? -1 : 0;
  } catch (...) { return -1; }
}
