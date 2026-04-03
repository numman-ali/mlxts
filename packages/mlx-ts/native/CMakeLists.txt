cmake_minimum_required(VERSION 3.24)
project(mlx-ts-native LANGUAGES C CXX)

include(FetchContent)

# mlx-c: Apple's official C API for MLX.
# mlx-c's own CMakeLists.txt uses FetchContent to pull MLX automatically.
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
  GIT_TAG v0.6.0
)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

FetchContent_MakeAvailable(mlx-c)
