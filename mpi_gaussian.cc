#include "mpi.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <list>
#include <stdio.h>
#include <iostream>

#include <El.hpp>

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::sort;
using std::string;
using std::normal_distribution;

const int N = 6; // dataset size
const int d = 2; // parameter dimension
const N_SAMPLES = 1000; // number of samples


void master_loop() {
  std::cout << " master" << std::endl;
}

template<typename Field, typename T>
El::Matrix<Field> sgldEstimate(const El::Matrix<Field>& theta, const El::DistMatrix<T>& X) {
  El::Matrix<Field> sgldEstimate(theta.Height(), theta.Width(), true);
  El::Zeros(sgldEstimate, theta.Height(), theta.Width());

  auto miniBatch = X.LockedMatrix();
  El::Matrix<> ones;
  El::Ones(ones, X.Width(), 1);

  // miniBatch -= theta * ones^T, each column is x_i - theta
  El::Ger(Field(-1), theta, ones, miniBatch);

  // *(1/miniBatch size), sum over cols, result is 1/n \sum_i x_i - \theta
  El::Gemv(El::NORMAL, Field(1.0 / miniBatch.Width()), miniBatch, ones, sgldEstimate);

  return sgldEstimate;
}

template<typename Field>
El::Matrix<Field> nablaLogPrior(const El::Matrix<Field>& theta) {
  El::Matrix<Field> nablaLogPrior;

  // Gaussian prior centered at origin, should be -\theta / \sigma^2
  nablaLogPrior = theta;
  nablaLogPrior *= -1.0;
  nablaLogPrior *= 1.0/10.0; // spread it out, equiv to making variance large

  return nablaLogPrior;
}

template<typename Field, typename T>
void sgldUpdate(const double& epsilon, El::Matrix<Field>& theta, const El::DistMatrix<T>& X) {
  auto theta0 = theta; // make copy of original value

  // Gradient of log prior
  El::Axpy(Field(epsilon / 2.0), nablaLogPrior(theta0), theta);

  // SGLD estimator
  El::Axpy(Field(epsilon / 2.0 * N), sgldEstimate(theta0, X), theta);

  // Injected Gaussian noise
  El::Matrix<Field> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());
  El::Axpy(Field(epsilon), nu, theta);
}

template<typename T>
void worker_loop(const int& myid, const El::DistMatrix<T>& X) {
  const double a = 2;

  El::Matrix<double> theta;
  El::Zeros(theta, d, 1);

  for (int i = 0; i < N_SAMPLES; ++i) {
    double epsilon = a / (i+1); // step size
    sgldUpdate(epsilon, theta, X);
    El::Print(theta);
  }
}

int main(int argc, char** argv) {
  int myid, pnum;
  try {
    El::Environment env(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    const bool is_master = myid == 0;

    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_master ? 0 : 1, myid, &worker_comm);
    El::Grid grid(worker_comm, 1, El::COLUMN_MAJOR); // one row, distribute columns

    // TODO: load the data up from disk
    // TODO(later): use alchemist to load pre-processed data form spark
    // For compatibility with BLAS, local matrices are column major. Hence, we store
    // one instance per column and distribute the matrix by columns
    El::DistMatrix<> X(d, N, grid);

    // example where we sample N/2 points from a standard normal centered at +2 and the rest from one centered at -2
    El::Gaussian(X, d, N);
    El::DistMatrix<> offsets(d, N, grid);
    El::Ones(offsets, X.Height(), N/2);
    offsets *= 2;
    X(El::ALL, El::IR(0, N/2)) += offsets;
    X(El::ALL, El::IR(N/2, X.Width())) -= offsets;

    if (is_master)
      master_loop();
    else
      worker_loop(myid, X);

  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
