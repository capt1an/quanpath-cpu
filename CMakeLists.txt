# 指定CMake的最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(QuanPath-cpp LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 指定源文件目录和头文件目录
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# 将头文件目录添加到包含路径中
include_directories(${INCLUDE_DIR})

# 搜索src目录下的所有源文件
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SRC_FILES})