#include <algorithm>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string_view>

#include "mlx/c/error.h"
#include "mlx/c/map.h"
#include "mlx/c/ops.h"
#include "mlx/c/private/map.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/string.h"
#include "mlx/c/string.h"
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

namespace {

using mlx::core::GGUFMetaData;
using mlx::core::array;

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
