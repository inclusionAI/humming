#pragma once

#include <humming/utils/all.cuh>


enum class LayoutType : uint32_t {
  Normal = 0,
  Grouped = 1,
  Permute = 2,
  GroupedPadded = 3,
  Scatter = 4,
};


enum class WorkPartition : uint32_t {
  Row = 0,
  Tile = 1,
};


struct InputLayoutParams {
  // Grouped/permute: E + 1 expert row offsets.
  // GroupedPadded: E valid-token counts.
  const void *expert_layout;
  // Permute: output row -> input row.
  // Scatter: input row * kScatterWidth + route -> output row.
  const void *indices;
  uint64_t num_input_rows;
  uint64_t num_output_rows;
  uint32_t num_experts;
  uint32_t max_tokens_per_expert;
};


template <
    LayoutType kType_,
    uint32_t kHiddenSize_,
    uint32_t kThreadsPerTask_,
    uint32_t kValuesPerThread_,
    uint32_t kTokensPerBlock_ = 1,
    uint32_t kScatterWidth_ = 1,
    bool kScatterSingleOutput_ = false,
    bool kDirectScatter_ = kScatterSingleOutput_,
    bool kExpertLayoutInt64_ = false,
    bool kIndexInt64_ = false,
    bool kZeroInvalid_ = false,
    WorkPartition kPartition_ = WorkPartition::Row,
    uint32_t kTileSize_ = kHiddenSize_,
    uint32_t kTilesPerBlock_ = 1>
class InputLayout {
public:
  static constexpr LayoutType kType = kType_;
  static constexpr uint32_t kHiddenSize = kHiddenSize_;
  static constexpr uint32_t kThreadsPerTask = kThreadsPerTask_;
  static constexpr uint32_t kValuesPerThread = kValuesPerThread_;
  static constexpr uint32_t kTokensPerBlock = kTokensPerBlock_;
  static constexpr uint32_t kScatterWidth = kScatterWidth_;
  static constexpr bool kScatterSingleOutput = kScatterSingleOutput_;
  static constexpr bool kDirectScatter = kType == LayoutType::Scatter && kTokensPerBlock == 1 && kDirectScatter_;
  static constexpr bool kExpertLayoutInt64 = kExpertLayoutInt64_;
  static constexpr bool kIndexInt64 = kIndexInt64_;
  static constexpr bool kZeroInvalid = kZeroInvalid_;
  static constexpr WorkPartition kPartition = kPartition_;
  static constexpr uint32_t kTileSize = kTileSize_;
  static constexpr uint32_t kTilesPerBlock = kTilesPerBlock_;
  static constexpr uint32_t kNumTiles = kHiddenSize / kTileSize;
  static constexpr uint32_t kTilesPerTask = kTilesPerBlock < kNumTiles ? kTilesPerBlock : kNumTiles;
  static constexpr uint32_t kColumnsPerTask = kTilesPerTask * kTileSize;
  static constexpr uint32_t kBlocksPerRow = kPartition == WorkPartition::Tile ? (kNumTiles + kTilesPerBlock - 1) / kTilesPerBlock : 1;
  static constexpr uint32_t kThreads = kThreadsPerTask * kTokensPerBlock;
  static constexpr uint32_t kOutputsPerToken = kType == LayoutType::Scatter && !kScatterSingleOutput ? kScatterWidth : 1;
  static constexpr bool kUniformRoutes = kScatterSingleOutput || kValuesPerThread <= 8;
  static constexpr uint32_t kRouteChunks = kDirectScatter && !kUniformRoutes ? (kOutputsPerToken + 31) / 32 : kOutputsPerToken;
  static constexpr uint64_t kInvalidRow = ~uint64_t{0};
  static constexpr bool kFullColumns =
      kPartition == WorkPartition::Row
          ? kThreadsPerTask * kValuesPerThread == kHiddenSize
          : kNumTiles % kTilesPerBlock == 0 && kThreadsPerTask * kValuesPerThread == kColumnsPerTask;

  static_assert(kHiddenSize > 0);
  static_assert(kThreadsPerTask > 0);
  static_assert(kValuesPerThread > 0);
  static_assert(kTokensPerBlock > 0);
  static_assert(kTileSize > 0 && kHiddenSize % kTileSize == 0);
  static_assert(kTilesPerBlock > 0);
  static_assert(kPartition != WorkPartition::Row || kThreadsPerTask * kValuesPerThread >= kHiddenSize);
  static_assert(kPartition != WorkPartition::Tile || kTokensPerBlock == 1);
  static_assert(kPartition != WorkPartition::Tile || kThreadsPerTask * kValuesPerThread >= kColumnsPerTask);
  static_assert(kType != LayoutType::Scatter || kScatterWidth > 0);
  static_assert(kType == LayoutType::Scatter || kScatterWidth == 1);
  static_assert(kType != LayoutType::Scatter || kIndexInt64);
  static_assert(!kScatterSingleOutput || kType == LayoutType::Scatter);
  static_assert(!kDirectScatter_ || kType == LayoutType::Scatter);
  static_assert(!kZeroInvalid || kType == LayoutType::GroupedPadded);

  struct BlockTask {
    uint64_t first_token;
    uint32_t num_tokens;
    uint32_t first_column;
    uint32_t num_columns;
  };

  struct TokenTask {
    uint64_t input_row;
    uint64_t output_rows[kOutputsPerToken];
    uint32_t expert;
    bool active;
    bool load;
    bool zero;
  };

  struct ThreadTask {
    uint64_t input_row;
    uint64_t element_offset;
    uint32_t token_in_block;
    uint32_t lane;
    uint32_t column;
    uint32_t num_values;
    uint32_t expert;
    bool load;

    CUDA_INLINE uint64_t input_offset() const {
      return element_offset;
    }

    template <class SourceType>
    CUDA_INLINE const SourceType *input(const SourceType *base) const {
      return base + input_offset();
    }
  };

  struct WriteTask {
    uint64_t output_row;
    uint64_t element_offset;
    uint32_t column;
    uint32_t num_values;
    bool write;
    bool zero;

    template <uint32_t kBits>
    CUDA_INLINE uint64_t output_byte_offset() const {
      static_assert(kHiddenSize * kBits % 8 == 0);
      static_assert(kValuesPerThread * kBits % 8 == 0);
      return element_offset * kBits / 8;
    }

    template <uint32_t kBits>
    CUDA_INLINE uint8_t *output(void *base) const {
      return reinterpret_cast<uint8_t *>(base) + output_byte_offset<kBits>();
    }
  };

  struct RouteTask {
    uint64_t rows[kRouteChunks];
    bool zero;
  };

  struct SharedStorage {
    TokenTask tokens[kTokensPerBlock];
  };

  CUDA_INLINE InputLayout(const InputLayoutParams &params, SharedStorage *shared)
      : params_(params), shared_(shared), first_token_(0), first_column_(0), last_column_(kHiddenSize),
        direct_input_row_(0), direct_route_(0) {
    if constexpr (kPartition == WorkPartition::Row) {
      first_token_ = static_cast<uint64_t>(blockIdx.x) * kTokensPerBlock;
    } else {
      uint64_t task = blockIdx.x / kBlocksPerRow;
      first_token_ = task;
      uint32_t first_tile = blockIdx.x % kBlocksPerRow * kTilesPerBlock;
      first_column_ = first_tile * kTileSize;
      last_column_ = min(first_column_ + kColumnsPerTask, kHiddenSize);
    }
    if constexpr (kDirectScatter) {
      direct_input_row_ = kScatterSingleOutput ? first_token_ / kScatterWidth : first_token_;
      direct_route_ = kScatterSingleOutput ? first_token_ % kScatterWidth : 0;
    }
  }

  __host__ __device__ static constexpr uint64_t grid_size(uint64_t rows) {
    if constexpr (kPartition == WorkPartition::Tile) {
      return rows * kBlocksPerRow;
    } else {
      return (rows + kTokensPerBlock - 1) / kTokensPerBlock;
    }
  }

  CUDA_INLINE BlockTask block() const {
    uint64_t rows = num_work_rows();
    uint64_t remaining = first_token_ < rows ? rows - first_token_ : 0;
    uint32_t max_tokens = kPartition == WorkPartition::Row ? kTokensPerBlock : 1;
    uint32_t num_tokens = static_cast<uint32_t>(remaining < max_tokens ? remaining : max_tokens);
    return BlockTask{first_token_, num_tokens, first_column_, last_column_ - first_column_};
  }

  // Must be called by every thread in the CTA before token/thread/write.
  // Normal layout has no metadata and compiles this method away.
  CUDA_INLINE void prepare() const {
    if constexpr (kType != LayoutType::Normal && !kDirectScatter) {
      for (uint32_t token = threadIdx.x; token < kTokensPerBlock; token += blockDim.x)
        shared_->tokens[token] = map_token(first_token_ + token);
      __syncthreads();
    }
  }

  CUDA_INLINE TokenTask token(uint32_t token_in_block) const {
    if constexpr (kType == LayoutType::Normal || kDirectScatter) {
      return map_token(first_token_ + token_in_block);
    } else {
      return shared_->tokens[token_in_block];
    }
  }

  CUDA_INLINE ThreadTask thread() const {
    uint32_t thread = threadIdx.x;
    uint32_t token_in_block = kPartition == WorkPartition::Row ? thread / kThreadsPerTask : 0;
    uint32_t lane = kPartition == WorkPartition::Row ? thread - token_in_block * kThreadsPerTask : thread;
    uint32_t column = first_column_ + lane * kValuesPerThread;
    uint32_t num_values = kFullColumns ? kValuesPerThread : (column < last_column_ ? min(kValuesPerThread, last_column_ - column) : 0);
    if constexpr (kType == LayoutType::Normal && kFullColumns && (kPartition == WorkPartition::Tile || kTokensPerBlock == 1)) {
      uint64_t row = first_token_;
      uint64_t element_offset = static_cast<uint64_t>(blockIdx.x) * (kPartition == WorkPartition::Tile ? kColumnsPerTask : kHiddenSize) + lane * kValuesPerThread;
      return ThreadTask{row, element_offset, token_in_block, lane, column, num_values, 0, true};
    }
    if constexpr (kDirectScatter) {
      bool load = num_values != 0;
      uint64_t element_offset = load ? direct_input_row_ * kHiddenSize + column : 0;
      return ThreadTask{load ? direct_input_row_ : 0, element_offset, token_in_block, lane, column, num_values, 0, load};
    }
    TokenTask task = token(token_in_block);
    bool load = task.load && num_values != 0;
    return ThreadTask{
        load ? task.input_row : 0,
        load ? task.input_row * kHiddenSize + column : 0,
        token_in_block,
        lane,
        column,
        num_values,
        task.expert,
        load};
  }

  CUDA_INLINE RouteTask routes(const ThreadTask &thread) const {
    RouteTask routes{};
    if constexpr (kDirectScatter) {
      if constexpr (kUniformRoutes) {
        uint32_t first_route = kScatterSingleOutput ? direct_route_ : 0;
        PRAGMA_UNROLL
        for (uint32_t route = 0; route < kOutputsPerToken; route++) {
          int64_t output_row = load_index<kIndexInt64>(params_.indices, direct_input_row_ * kScatterWidth + first_route + route);
          routes.rows[route] = static_cast<uint64_t>(output_row);
        }
      } else {
        uint32_t lane = threadIdx.x & 31;
        PRAGMA_UNROLL
        for (uint32_t chunk = 0; chunk < kRouteChunks; chunk++) {
          uint32_t route = chunk * 32 + lane;
          int64_t output_row = route < kScatterWidth ? load_index<kIndexInt64>(params_.indices, direct_input_row_ * kScatterWidth + route) : -1;
          routes.rows[chunk] = static_cast<uint64_t>(output_row);
        }
      }
    } else {
      TokenTask token_task = token(thread.token_in_block);
      PRAGMA_UNROLL
      for (uint32_t route = 0; route < kOutputsPerToken; route++)
        routes.rows[route] = token_task.output_rows[route];
      routes.zero = token_task.zero;
    }
    return routes;
  }

  CUDA_INLINE WriteTask write(const ThreadTask &thread, const RouteTask &routes, uint32_t route = 0) const {
    if constexpr (kType == LayoutType::Normal && kFullColumns && (kPartition == WorkPartition::Tile || kTokensPerBlock == 1))
      return WriteTask{thread.input_row, thread.element_offset, thread.column, thread.num_values, true, false};
    if constexpr (kDirectScatter) {
      uint64_t output_row;
      if constexpr (kUniformRoutes) output_row = routes.rows[route];
      else output_row = __shfl_sync(0xFFFFFFFFu, routes.rows[route / 32], route & 31);
      bool write = thread.num_values != 0 && output_row < params_.num_output_rows;
      uint64_t row = write ? output_row : 0;
      return WriteTask{row, write ? row * kHiddenSize + thread.column : 0, thread.column, thread.num_values, write, false};
    }
    uint64_t output_row = route < kOutputsPerToken ? routes.rows[route] : kInvalidRow;
    bool write = thread.num_values != 0 && output_row != kInvalidRow;
    return WriteTask{write ? output_row : 0, write ? output_row * kHiddenSize + thread.column : 0, thread.column, thread.num_values, write, write && routes.zero};
  }

private:
  template <bool kInt64>
  CUDA_INLINE static int64_t load_index(const void *pointer, uint64_t index) {
    if constexpr (kInt64) {
      return reinterpret_cast<const int64_t *>(pointer)[index];
    } else {
      return reinterpret_cast<const int32_t *>(pointer)[index];
    }
  }

  CUDA_INLINE uint64_t num_work_rows() const {
    if constexpr (kType == LayoutType::Scatter) {
      return params_.num_input_rows * (kScatterSingleOutput ? kScatterWidth : 1);
    } else {
      return params_.num_output_rows;
    }
  }

  CUDA_INLINE uint32_t find_expert(uint64_t row) const {
    uint32_t lo = 0;
    uint32_t hi = params_.num_experts;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      uint64_t end = static_cast<uint64_t>(load_index<kExpertLayoutInt64>(params_.expert_layout, mid + 1));
      if (row >= end) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  CUDA_INLINE TokenTask map_token(uint64_t logical_row) const {
    TokenTask task{};
    PRAGMA_UNROLL
    for (uint32_t route = 0; route < kOutputsPerToken; route++)
      task.output_rows[route] = kInvalidRow;
    if constexpr (kType == LayoutType::Normal && (kPartition == WorkPartition::Tile || kTokensPerBlock == 1)) {
      task.active = true;
    } else {
      task.active = logical_row < num_work_rows();
    }
    if (!task.active) return task;

    if constexpr (kType == LayoutType::Normal) {
      task.input_row = logical_row;
      if constexpr (kPartition == WorkPartition::Tile || kTokensPerBlock == 1) {
        task.load = true;
      } else {
        task.load = logical_row < params_.num_input_rows;
      }
      if (task.load) task.output_rows[0] = logical_row;
    } else if constexpr (kType == LayoutType::Grouped || kType == LayoutType::Permute) {
      task.expert = find_expert(logical_row);
      if (task.expert >= params_.num_experts) return task;
      if constexpr (kType == LayoutType::Permute) {
        task.input_row = static_cast<uint64_t>(load_index<kIndexInt64>(params_.indices, logical_row));
      } else {
        task.input_row = logical_row;
      }
      task.load = task.input_row < params_.num_input_rows;
      if (task.load) task.output_rows[0] = logical_row;
    } else if constexpr (kType == LayoutType::GroupedPadded) {
      task.expert = static_cast<uint32_t>(logical_row / params_.max_tokens_per_expert);
      if (task.expert >= params_.num_experts) return task;
      uint32_t local_row = static_cast<uint32_t>(logical_row % params_.max_tokens_per_expert);
      uint64_t valid_tokens = static_cast<uint64_t>(load_index<kExpertLayoutInt64>(params_.expert_layout, task.expert));
      task.load = local_row < valid_tokens && logical_row < params_.num_input_rows;
      task.input_row = task.load ? logical_row : 0;
      if (task.load || kZeroInvalid) task.output_rows[0] = logical_row;
      task.zero = !task.load && kZeroInvalid;
    } else {
      uint32_t first_route = 0;
      if constexpr (kScatterSingleOutput) {
        task.input_row = logical_row / kScatterWidth;
        first_route = logical_row % kScatterWidth;
      } else {
        task.input_row = logical_row;
      }
      bool has_output = false;
      uint32_t last_route = kScatterSingleOutput ? first_route + 1 : kScatterWidth;
      PRAGMA_UNROLL
      for (uint32_t route = first_route; route < last_route; route++) {
        int64_t output_row = -1;
        if constexpr (kDirectScatter) {
          if ((threadIdx.x & 31) == 0)
            output_row = load_index<kIndexInt64>(params_.indices, task.input_row * kScatterWidth + route);
          output_row = __shfl_sync(0xFFFFFFFFu, output_row, 0);
        } else {
          output_row = load_index<kIndexInt64>(params_.indices, task.input_row * kScatterWidth + route);
        }
        uint64_t row = static_cast<uint64_t>(output_row);
        if (row < params_.num_output_rows) {
          task.output_rows[kScatterSingleOutput ? 0 : route] = row;
          has_output = true;
        }
      }
      task.load = has_output;
    }
    return task;
  }

  InputLayoutParams params_;
  SharedStorage *shared_;
  uint64_t first_token_;
  uint32_t first_column_;
  uint32_t last_column_;
  uint64_t direct_input_row_;
  uint32_t direct_route_;
};
