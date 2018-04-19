cmake_minimum_required(VERSION 3.1)
project(TravellingClusterChain)

# Set C++11
set (CMAKE_CXX_STANDARD 11)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

# Require MPI for this project:
find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAG "${CMAKE_CXX_FLAG} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(MPI_INCLUDE_PATH)

# Add Elemental
add_subdirectory(external/Elemental)
include_directories("${PROJECT_BINARY_DIR}/external/El/include")
include_directories(${MPI_CXX_INCLUDE_PATH})

# Define sampler library
add_library(sampler STATIC sampler.cc)
target_link_libraries(sampler ${MPI_LIBRARIES} El)

# Define experiments
add_executable(mpi_gaussian_imbalance
    mpi_gaussian_imbalance.cc)
target_link_libraries(mpi_gaussian_imbalance sampler)

