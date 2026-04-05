#include <cmath>

#include "mlx/c/error.h"
#include "mlx/c/ops.h"
#include "mlx/c/private/mlx.h"

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
