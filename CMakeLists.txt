
cmake_minimum_required(VERSION 3.18)

project(TestCompilation LANGUAGES CUDA CXX)

# Checking CUDA availability
include(CheckLanguage)
check_language(CUDA)

# Set C++ 17 as the standard to use
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:\"LIBCMT\"")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})

add_library(base)

target_sources(base
  PRIVATE
    src/cpp_TP1.cpp
    src/cpp_TP3_glob.cpp
    src/cpp_TP3_shrd.cpp
)
target_sources(base
  PRIVATE
    src/cuda_TP1.cu
    src/cuda_TP3_glob.cu
    src/cuda_TP3_shrd.cu
)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    base
)