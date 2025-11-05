# cmake_minimum_required(VERSION 3.12)
# project(LBM_Kokkos CXX)

# # 设置C++标准
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 查找Kokkos库
# include(cmake/build_or_find_kokkos.cmake)

# # 添加可执行文件
# add_executable(lbm_simulation 
#     src/Boltzmann_harmonic3D.cpp
#     src/main.cpp
#     src/configs.cpp
# )

# # 链接Kokkos库
# target_link_libraries(lbm_simulation Kokkos::kokkos)

# # 设置编译器选项
# target_compile_options(lbm_simulation PRIVATE -Wall -Wextra -fPIE)


# # 包含目录
# target_include_directories(lbm_simulation PRIVATE 
#     ${CMAKE_SOURCE_DIR}/include
#     ${Kokkos_INCLUDE_DIRS}
# )

# # 性能优化选项
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     target_compile_options(lbm_simulation PRIVATE -O3)
# endif()

cmake_minimum_required(VERSION 3.18)
project(Boltzmann_harmonic3D LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

# 1. Kokkos
include(cmake/build_or_find_kokkos.cmake)

# 2. Lua（核心修复：简化头文件路径配置）
# --- Lua 5.4.6 ---
set(LUA_SRC_DIR "${EXTERNAL_DIR}/Lua/lua-5.4.6/src")
set(LUA_INC_DIR "${EXTERNAL_DIR}/Lua/lua-5.4.6/include")

file(GLOB LUA_SOURCES "${LUA_SRC_DIR}/*.c")
list(FILTER LUA_SOURCES EXCLUDE REGEX "lua.c|luac.c")

add_library(lua STATIC ${LUA_SOURCES})
set_target_properties(lua PROPERTIES LINKER_LANGUAGE C)

# 1. 让 *.c 找到 lua.h
target_include_directories(lua
    PRIVATE ${LUA_INC_DIR}   # 编译期
    PUBLIC  ${LUA_INC_DIR}   # 给别人用
)

# 3. sol2
set(SOL2_DIR "${EXTERNAL_DIR}/sol2")
add_library(sol2 INTERFACE)
target_include_directories(sol2 INTERFACE ${SOL2_DIR})

# 4. spdlog
find_package(spdlog REQUIRED CONFIG)

# 5. 可执行文件
add_executable(boltzmann_harmonic3d
    src/main.cpp
    src/configs.cpp
    src/Boltzmann_harmonic3D.cpp
)

# 6. 链接依赖
target_link_libraries(boltzmann_harmonic3d PRIVATE
    Kokkos::kokkos
    lua
    sol2
    spdlog::spdlog
)

# 7. 让任何链接 boltzmann_harmonic3d 的目标都能找到头文件
target_include_directories(boltzmann_harmonic3d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# 8. OpenMP
if(KOKKOS_LBM_BACKEND STREQUAL "OpenMP")
    target_compile_options(boltzmann_harmonic3d PRIVATE -fopenmp)
    target_link_options(boltzmann_harmonic3d PRIVATE -fopenmp)
endif()
