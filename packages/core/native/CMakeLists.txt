cmake_minimum_required(VERSION 3.24)
project(mlxts-core-native LANGUAGES C CXX)

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

add_library(
  mlxts_core_native SHARED
  ${CMAKE_CURRENT_LIST_DIR}/mlxts_core_ops.cpp
)

target_link_libraries(mlxts_core_native PRIVATE mlxc mlx)
set_property(TARGET mlxts_core_native PROPERTY CXX_STANDARD 20)
set_property(TARGET mlxts_core_native PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories(
  mlxts_core_native
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)
set_property(TARGET mlxts_core_native PROPERTY POSITION_INDEPENDENT_CODE ON)
