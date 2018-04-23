cmake_minimum_required(VERSION 3.1)

project(TravellingClusterChain)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

# Require MPI for this project:
find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAG "${CMAKE_CXX_FLAG} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})

# Add GSL
set(GSL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/gsl)
set(GSL_BINARY_DIR ${PROJECT_BINARY_DIR}/external/gsl)
ExternalProject_Add(gsl
  PREFIX          ${CMAKE_INSTALL_PREFIX}
  GIT_REPOSITORY  https://github.com/ampl/gsl
  GIT_TAG         v2.4.0
  STAMP_DIR       ${GSL_BINARY_DIR}/stamp
  SOURCE_DIR      ${GSL_SOURCE_DIR}
  BINARY_DIR      ${GSL_BINARY_DIR}
  TMP_DIR         ${GSL_BINARY_DIR}/tmp
  INSTALL_COMMAND ""
  UPDATE_COMMAND  ""
  CMAKE_ARGS
      -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -D CMAKE_C_FLAGS=${CMAKE_C_FLAGS}
      -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -D CMAKE_EXECUTABLE_SUFFIX_C=${CMAKE_EXECUTABLE_SUFFIX_C}
      -D CMAKE_MACOSX_RPATH=${CMAKE_MACOSX_RPATH}
      -D CMAKE_SKIP_RPATH=${CMAKE_SKIP_RPATH}
      -D CMAKE_SKIP_BUILD_RPATH=${CMAKE_SKIP_BUILD_RPATH}
      -D CMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH}
      -D CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
      -D GSL_INSTALL=TRUE
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  )
# TODO: install the library and define it as a target
include_directories(${PROJECT_BINARY_DIR}/external/gsl)
add_library(libgsl STATIC IMPORTED)
set_property(TARGET libgsl PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/external/gsl/libgsl.dylib)

# Add Elemental
add_subdirectory(external/Elemental)
include_directories("${PROJECT_BINARY_DIR}/external/Elemental/include")
include_directories("${PROJECT_SOURCE_DIR}/external/Elemental/include")
include_directories("${PROJECT_SOURCE_DIR}/external/Elemental/external/suite_sparse/include")

#set(LIBS ${MPI_LIBRARIES} El)

# Our libraries
add_library(sampler STATIC sampler.cc)
target_link_libraries(sampler El)

add_library(sgld_sampler STATIC sgld_sampler.cc)
target_link_libraries(sgld_sampler sampler)

add_library(sgrld_sampler STATIC sgrld_sampler.cc)
target_link_libraries(sgrld_sampler sampler)

add_library(gmm_toy_model STATIC gmm_toy_model.cc)
add_dependencies(gmm_toy_model gsl)
target_link_libraries(gmm_toy_model El libgsl)

add_library(lda_model STATIC lda_model.cc)
add_dependencies(lda_model gsl)
target_link_libraries(lda_model El libgsl)

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
