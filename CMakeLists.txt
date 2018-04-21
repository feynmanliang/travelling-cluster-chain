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

# Add GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})


set(LIBS ${LIBS} ${GSL_LIBRARIES} ${MPI_LIBRARIES} El)

add_library(sampler STATIC sampler.cc)
target_link_libraries(sampler ${LIBS})

add_library(sgld_sampler STATIC sgld_sampler.cc)
target_link_libraries(sgld_sampler sampler)

add_library(sgrld_sampler STATIC sgrld_sampler.cc)
target_link_libraries(sgrld_sampler sampler)

add_library(gmm_toy_model STATIC gmm_toy_model.cc)
add_library(lda_model STATIC lda_model.cc)

# Define experiments
add_executable(mpi_gaussian
    mpi_gaussian.cc)
target_link_libraries(mpi_gaussian sgld_sampler gmm_toy_model)

add_executable(mpi_gaussian_imbalance
    mpi_gaussian_imbalance.cc)
target_link_libraries(mpi_gaussian_imbalance sgld_sampler gmm_toy_model)

add_executable(mpi_lda_sgrld
    mpi_lda_sgrld.cc)
target_link_libraries(mpi_lda_sgrld sgrld_sampler lda_model)

add_executable(mpi_lda_testdata
    mpi_lda_testdata.cc)
target_link_libraries(mpi_lda_testdata sgrld_sampler lda_model)
