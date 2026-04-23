#include <cmath>
#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

#include "mlx/c/error.h"
#include "mlx/c/map.h"
#include "mlx/c/ops.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/map.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/vector.h"
#include "mlx/c/private/string.h"
#include "mlx/c/string.h"
#include "mlx/fast.h"
#include "mlx/io.h"

extern "C" int
mlxts_gelu_approx(mlx_array* res, const mlx_array a, const mlx_stream s) {
  try {
    auto& input = mlx_array_get_(a);
    auto& stream = mlx_stream_get_(s);
    using mlx::core::add;
    using mlx::core::array;
    using mlx::core::multiply;
    using mlx::core::tanh;

    auto cube_scale = array(0.044715f, input.dtype());
    auto half = array(0.5f, input.dtype());
    auto one = array(1.0f, input.dtype());
    auto tanh_scale = array(
        static_cast<float>(std::sqrt(2.0 / 3.14159265358979323846)),
        input.dtype());

    auto input_squared = multiply(input, input, stream);
    auto input_cubed = multiply(input_squared, input, stream);
    auto scaled_cube = multiply(input_cubed, cube_scale, stream);
    auto inner = add(input, scaled_cube, stream);
    auto scaled_inner = multiply(inner, tanh_scale, stream);
    auto activated = tanh(scaled_inner, stream);
    auto shifted = add(activated, one, stream);
    auto scaled_input = multiply(input, half, stream);
    mlx_array_set_(*res, multiply(scaled_input, shifted, stream));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_slice_update_inplace(
    const mlx_array src,
    const mlx_array update,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) {
  try {
    auto updated = mlx::core::slice_update(
        mlx_array_get_(src),
        mlx_array_get_(update),
        mlx::core::Shape(start, start + start_num),
        mlx::core::Shape(stop, stop + stop_num),
        mlx::core::Shape(strides, strides + strides_num),
        mlx_stream_get_(s));
    auto mutable_src = src;
    mlx_array_set_(mutable_src, std::move(updated));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_array_assign_inplace(
    const mlx_array target,
    const mlx_array source) {
  try {
    auto mutable_target = target;
    mlx_array_set_(mutable_target, mlx_array_get_(source));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_slice_view_inplace(
    const mlx_array target,
    const mlx_array source,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) {
  try {
    auto view = mlx::core::slice(
        mlx_array_get_(source),
        mlx::core::Shape(start, start + start_num),
        mlx::core::Shape(stop, stop + stop_num),
        mlx::core::Shape(strides, strides + strides_num),
        mlx_stream_get_(s));
    auto mutable_target = target;
    mlx_array_set_(mutable_target, std::move(view));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

namespace {

using mlx::core::GGUFMetaData;
using mlx::core::array;

const mlx::core::fast::CustomKernelFunction& qwen_gated_delta_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "mlxts_qwen_gated_delta_update",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      R"metal(
        const uint combined_head = thread_position_in_grid.z;
        const uint batch = combined_head / Hv;
        const uint value_head = combined_head % Hv;
        const uint key_head = value_head / (Hv / Hk);
        const uint value_dim_index = thread_position_in_grid.y;
        const uint lane = thread_position_in_threadgroup.x;
        constexpr int values_per_lane = Dk / 32;

        auto q_row = q + ((batch * T * Hk + key_head) * Dk);
        auto k_row = k + ((batch * T * Hk + key_head) * Dk);
        auto v_row = v + ((batch * T * Hv + value_head) * Dv);
        auto g_row = g + (batch * T * Hv);
        auto beta_row = beta + (batch * T * Hv);
        auto state_row = state_in + ((combined_head * Dv + value_dim_index) * Dk);
        auto next_state_row = state_out + ((combined_head * Dv + value_dim_index) * Dk);
        auto y_row = y + ((batch * T * Hv + value_head) * Dv);

        float local_state[values_per_lane];
        for (int offset = 0; offset < values_per_lane; ++offset) {
          const uint key_dim_index = values_per_lane * lane + offset;
          local_state[offset] = static_cast<float>(state_row[key_dim_index]);
        }

        for (int step = 0; step < T; ++step) {
          float kv_memory = 0.0f;
          for (int offset = 0; offset < values_per_lane; ++offset) {
            const uint key_dim_index = values_per_lane * lane + offset;
            local_state[offset] *= static_cast<float>(g_row[value_head]);
            kv_memory += local_state[offset] * static_cast<float>(k_row[key_dim_index]);
          }
          kv_memory = simd_sum(kv_memory);

          const float delta =
              (static_cast<float>(v_row[value_dim_index]) - kv_memory) *
              static_cast<float>(beta_row[value_head]);

          float output_value = 0.0f;
          for (int offset = 0; offset < values_per_lane; ++offset) {
            const uint key_dim_index = values_per_lane * lane + offset;
            local_state[offset] += static_cast<float>(k_row[key_dim_index]) * delta;
            output_value += local_state[offset] * static_cast<float>(q_row[key_dim_index]);
          }
          output_value = simd_sum(output_value);

          if (thread_index_in_simdgroup == 0) {
            y_row[value_dim_index] = static_cast<InT>(output_value);
          }

          q_row += Hk * Dk;
          k_row += Hk * Dk;
          v_row += Hv * Dv;
          g_row += Hv;
          beta_row += Hv;
          y_row += Hv * Dv;
        }

        for (int offset = 0; offset < values_per_lane; ++offset) {
          const uint key_dim_index = values_per_lane * lane + offset;
          next_state_row[key_dim_index] = static_cast<StT>(local_state[offset]);
        }
      )metal");
  return kernel;
}

void expect_rank(const array& value, int rank, const char* name) {
  if (static_cast<int>(value.ndim()) != rank) {
    std::ostringstream message;
    message << "mlxts_qwen_gated_delta_update: expected " << name << " rank "
            << rank << ", got rank " << value.ndim() << ".";
    throw std::invalid_argument(message.str());
  }
}

int shape_dim(const array& value, int axis) {
  return static_cast<int>(value.shape(axis));
}

void expect_dim(
    const array& value,
    int axis,
    int expected,
    const char* name,
    const char* dimension) {
  const auto actual = shape_dim(value, axis);
  if (actual != expected) {
    std::ostringstream message;
    message << "mlxts_qwen_gated_delta_update: expected " << name << " "
            << dimension << " to be " << expected << ", got " << actual << ".";
    throw std::invalid_argument(message.str());
  }
}

void append_json_escaped(std::ostringstream& out, std::string_view value) {
  out << '"';
  for (const char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          std::ostringstream escaped;
          escaped << "\\u" << std::hex << std::uppercase << std::setw(4)
                  << std::setfill('0')
                  << static_cast<int>(static_cast<unsigned char>(ch));
          out << escaped.str();
        } else {
          out << ch;
        }
        break;
    }
  }
  out << '"';
}

template <typename T>
void append_scalar_value(std::ostringstream& out, T value) {
  out << value;
}

template <>
void append_scalar_value<bool>(std::ostringstream& out, bool value) {
  out << (value ? "true" : "false");
}

template <>
void append_scalar_value<uint8_t>(std::ostringstream& out, uint8_t value) {
  out << static_cast<unsigned int>(value);
}

template <>
void append_scalar_value<int8_t>(std::ostringstream& out, int8_t value) {
  out << static_cast<int>(value);
}

template <typename T>
void append_array_values(std::ostringstream& out, const array& value) {
  auto data = value.data<T>();
  out << "[";
  for (int i = 0; i < value.size(); i++) {
    if (i > 0) {
      out << ",";
    }
    append_scalar_value<T>(out, data[i]);
  }
  out << "]";
}

void append_json_for_array(std::ostringstream& out, const array& value) {
  mlx::core::eval(value);

  const auto dtype = value.dtype();
  if (value.ndim() == 0) {
    switch (dtype) {
      case mlx::core::bool_:
        append_scalar_value<bool>(out, value.data<bool>()[0]);
        return;
      case mlx::core::uint8:
        append_scalar_value<uint8_t>(out, value.data<uint8_t>()[0]);
        return;
      case mlx::core::uint16:
        append_scalar_value<uint16_t>(out, value.data<uint16_t>()[0]);
        return;
      case mlx::core::uint32:
        append_scalar_value<uint32_t>(out, value.data<uint32_t>()[0]);
        return;
      case mlx::core::uint64:
        append_scalar_value<uint64_t>(out, value.data<uint64_t>()[0]);
        return;
      case mlx::core::int8:
        append_scalar_value<int8_t>(out, value.data<int8_t>()[0]);
        return;
      case mlx::core::int16:
        append_scalar_value<int16_t>(out, value.data<int16_t>()[0]);
        return;
      case mlx::core::int32:
        append_scalar_value<int32_t>(out, value.data<int32_t>()[0]);
        return;
      case mlx::core::int64:
        append_scalar_value<int64_t>(out, value.data<int64_t>()[0]);
        return;
      case mlx::core::float16:
        append_scalar_value<float>(out, static_cast<float>(value.data<mlx::core::float16_t>()[0]));
        return;
      case mlx::core::float32:
        append_scalar_value<float>(out, value.data<float>()[0]);
        return;
      default:
        throw std::runtime_error("mlxts_load_gguf: unsupported scalar metadata dtype.");
    }
  }

  if (value.ndim() != 1) {
    throw std::runtime_error("mlxts_load_gguf: only scalar and 1D metadata arrays are supported.");
  }

  switch (dtype) {
    case mlx::core::bool_:
      append_array_values<bool>(out, value);
      return;
    case mlx::core::uint8:
      append_array_values<uint8_t>(out, value);
      return;
    case mlx::core::uint16:
      append_array_values<uint16_t>(out, value);
      return;
    case mlx::core::uint32:
      append_array_values<uint32_t>(out, value);
      return;
    case mlx::core::uint64:
      append_array_values<uint64_t>(out, value);
      return;
    case mlx::core::int8:
      append_array_values<int8_t>(out, value);
      return;
    case mlx::core::int16:
      append_array_values<int16_t>(out, value);
      return;
    case mlx::core::int32:
      append_array_values<int32_t>(out, value);
      return;
    case mlx::core::int64:
      append_array_values<int64_t>(out, value);
      return;
    case mlx::core::float16:
      append_array_values<mlx::core::float16_t>(out, value);
      return;
    case mlx::core::float32:
      append_array_values<float>(out, value);
      return;
    default:
      throw std::runtime_error("mlxts_load_gguf: unsupported vector metadata dtype.");
  }
}

void append_json_for_metadata(std::ostringstream& out, const GGUFMetaData& value) {
  if (std::holds_alternative<std::monostate>(value)) {
    out << "null";
    return;
  }
  if (const auto* string_value = std::get_if<std::string>(&value); string_value != nullptr) {
    append_json_escaped(out, *string_value);
    return;
  }
  if (const auto* strings = std::get_if<std::vector<std::string>>(&value); strings != nullptr) {
    out << "[";
    for (size_t index = 0; index < strings->size(); index++) {
      if (index > 0) {
        out << ",";
      }
      append_json_escaped(out, strings->at(index));
    }
    out << "]";
    return;
  }
  if (const auto* array_value = std::get_if<array>(&value); array_value != nullptr) {
    append_json_for_array(out, *array_value);
    return;
  }

  throw std::runtime_error("mlxts_load_gguf: unsupported metadata variant.");
}

std::string serialize_metadata_json(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
  std::vector<std::string> keys;
  keys.reserve(metadata.size());
  for (const auto& [key, _] : metadata) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  std::ostringstream out;
  out << "{";
  bool first = true;
  for (const auto& key : keys) {
    if (!first) {
      out << ",";
    }
    first = false;
    append_json_escaped(out, key);
    out << ":";
    append_json_for_metadata(out, metadata.at(key));
  }
  out << "}";
  return out.str();
}

std::string join_array_map_keys(const mlx_map_string_to_array map) {
  auto& cpp_map = mlx_map_string_to_array_get_(map);
  std::vector<std::string> keys;
  keys.reserve(cpp_map.size());
  for (const auto& [key, _] : cpp_map) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  std::ostringstream out;
  for (size_t index = 0; index < keys.size(); index++) {
    if (index > 0) {
      out << "\n";
    }
    out << keys[index];
  }
  return out.str();
}

std::unordered_map<std::string, GGUFMetaData> metadata_from_string_map(
    const mlx_map_string_to_string metadata) {
  std::unordered_map<std::string, GGUFMetaData> converted;
  auto& cpp_map = mlx_map_string_to_string_get_(metadata);
  for (const auto& [key, value] : cpp_map) {
    converted.insert_or_assign(key, value);
  }
  return converted;
}

} // namespace

extern "C" int mlxts_qwen_gated_delta_update(
    mlx_array* output,
    mlx_array* state_out,
    const mlx_array q,
    const mlx_array k,
    const mlx_array v,
    const mlx_array g,
    const mlx_array beta,
    const mlx_array state,
    const mlx_stream s) {
  try {
    const auto& q_array = mlx_array_get_(q);
    const auto& k_array = mlx_array_get_(k);
    const auto& v_array = mlx_array_get_(v);
    const auto& g_array = mlx_array_get_(g);
    const auto& beta_array = mlx_array_get_(beta);
    const auto& state_array = mlx_array_get_(state);

    expect_rank(q_array, 4, "q");
    expect_rank(k_array, 4, "k");
    expect_rank(v_array, 4, "v");
    expect_rank(g_array, 3, "g");
    expect_rank(beta_array, 3, "beta");
    expect_rank(state_array, 4, "state");

    const auto batch_size = shape_dim(q_array, 0);
    const auto sequence_length = shape_dim(q_array, 1);
    const auto key_heads = shape_dim(q_array, 2);
    const auto key_head_dim = shape_dim(q_array, 3);
    const auto value_heads = shape_dim(v_array, 2);
    const auto value_head_dim = shape_dim(v_array, 3);

    expect_dim(k_array, 0, batch_size, "k", "batch");
    expect_dim(k_array, 1, sequence_length, "k", "sequence");
    expect_dim(k_array, 2, key_heads, "k", "key heads");
    expect_dim(k_array, 3, key_head_dim, "k", "key head dim");
    expect_dim(v_array, 0, batch_size, "v", "batch");
    expect_dim(v_array, 1, sequence_length, "v", "sequence");
    expect_dim(g_array, 0, batch_size, "g", "batch");
    expect_dim(g_array, 1, sequence_length, "g", "sequence");
    expect_dim(g_array, 2, value_heads, "g", "value heads");
    expect_dim(beta_array, 0, batch_size, "beta", "batch");
    expect_dim(beta_array, 1, sequence_length, "beta", "sequence");
    expect_dim(beta_array, 2, value_heads, "beta", "value heads");
    expect_dim(state_array, 0, batch_size, "state", "batch");
    expect_dim(state_array, 1, value_heads, "state", "value heads");
    expect_dim(state_array, 2, value_head_dim, "state", "value head dim");
    expect_dim(state_array, 3, key_head_dim, "state", "key head dim");

    if (key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0) {
      std::ostringstream message;
      message << "mlxts_qwen_gated_delta_update: value heads " << value_heads
              << " must be divisible by key heads " << key_heads << ".";
      throw std::invalid_argument(message.str());
    }
    if (key_head_dim <= 0 || key_head_dim % 32 != 0) {
      std::ostringstream message;
      message << "mlxts_qwen_gated_delta_update: key head dim "
              << key_head_dim << " must be a positive multiple of 32.";
      throw std::invalid_argument(message.str());
    }

    const auto input_dtype = q_array.dtype();

    std::vector<array> inputs = {
        q_array,
        k_array,
        v_array,
        g_array,
        beta_array,
        state_array,
        array(sequence_length, mlx::core::int32)};
    std::vector<mlx::core::Shape> output_shapes = {
        {batch_size, sequence_length, value_heads, value_head_dim},
        state_array.shape()};
    std::vector<mlx::core::Dtype> output_dtypes = {
        input_dtype,
        state_array.dtype()};
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        template_args = {
            {"InT", input_dtype},
            {"StT", state_array.dtype()},
            {"Dk", key_head_dim},
            {"Dv", value_head_dim},
            {"Hk", key_heads},
            {"Hv", value_heads}};

    auto outputs = qwen_gated_delta_kernel()(
        inputs,
        output_shapes,
        output_dtypes,
        {32, value_head_dim, batch_size * value_heads},
        {32, 4, 1},
        template_args,
        std::nullopt,
        false,
        mlx_stream_get_(s));
    if (outputs.size() != 2) {
      throw std::runtime_error(
          "mlxts_qwen_gated_delta_update: expected two native outputs.");
    }

    mlx_array_set_(*output, outputs[0]);
    mlx_array_set_(*state_out, outputs[1]);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_load_gguf(
    mlx_map_string_to_array* weights,
    mlx_string* metadata_json,
    const char* file,
    const mlx_stream s) {
  try {
    auto [loaded_weights, loaded_metadata] =
        mlx::core::load_gguf(std::string(file), mlx_stream_get_(s));
    mlx_map_string_to_array_set_(*weights, std::move(loaded_weights));
    mlx_string_set_(*metadata_json, serialize_metadata_json(loaded_metadata));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_save_gguf(
    const char* file,
    const mlx_map_string_to_array weights,
    const mlx_map_string_to_string metadata) {
  try {
    mlx::core::save_gguf(
        std::string(file),
        mlx_map_string_to_array_get_(weights),
        metadata_from_string_map(metadata));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxts_map_string_to_array_keys(
    mlx_string* out,
    const mlx_map_string_to_array map) {
  try {
    mlx_string_set_(*out, join_array_map_keys(map));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
