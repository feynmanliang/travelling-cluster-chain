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

const int N = 100; // dataset size
const int d = 2; // parameter dimension
const int N_SAMPLES = 100000; // number of samples


void master_loop() {
  std::cout << " master" << std::endl;
}

template <typename T>
T normal_pdf(T x, T m, T s) {
  static const T inv_sqrt_2pi = 0.3989422804014327;
  T a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}

template<typename Field, typename T>
El::Matrix<Field> sgldEstimate(const El::Matrix<Field>& theta, const El::DistMatrix<T>& X) {
  El::Matrix<Field> sgldEstimate(theta.Height(), theta.Width(), true);
  El::Zeros(sgldEstimate, theta.Height(), theta.Width());

  auto miniBatch = X.LockedMatrix();

  for (int i=0; i<miniBatch.Width(); ++i) {
    auto x = miniBatch(0, i);
    auto p0 = normal_pdf(x, theta(0), 2.0);
    auto p1 = normal_pdf(x, theta(0) + theta(1), 2.0);

    auto denom = p0 + p1;

    auto score0 = p0 / denom;
    auto score1 = p1 / denom;
    sgldEstimate(0, 0) += score0 * ((x - theta(0)) / 2.0);
    sgldEstimate(0, 0) += score1 * (x - theta(0) - theta(1)) / 2.0;
    sgldEstimate(1, 0) += score1 * (x - theta(0) - theta(1)) / 2.0;
  }

  sgldEstimate *= 1.0 / miniBatch.Width();

  return sgldEstimate;
}

template<typename Field>
El::Matrix<Field> nablaLogPrior(const El::Matrix<Field>& theta) {
  El::Matrix<Field> nablaLogPrior;

  // Gaussian prior centered at origin, should be -\theta / \sigma^2
  nablaLogPrior = theta;
  nablaLogPrior *= -1.0;
  nablaLogPrior(El::IR(0), 0) *= 0.1; // prior variance \sigma_1^2 = 10.0

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
  El::Axpy(El::Sqrt(epsilon), nu, theta);
}

template<typename T>
void worker_loop(const int& myid, const El::DistMatrix<T>& X) {
  // zero initialization
  El::Matrix<double> theta(d, 1);
  El::Ones(theta, d, 1);

  // TODO?: burn in before collecting samples

  El::Matrix<double> samples(d, N_SAMPLES);
  for (int i = 0; i < N_SAMPLES; ++i) {
    // TODO: exchange chains over different partitions, Send and Recv here

    // save a sample
    samples(El::ALL, i) = theta;

    double epsilon = 0.04 / El::Pow(10.0 + i, 0.55); // step size
    sgldUpdate(epsilon, theta, X);
  }

  // TODO: write samples to disk
  El::Write(samples, "samples-" + std::to_string(myid), El::MATRIX_MARKET);
}

int main(int argc, char** argv) {
  int myid, pnum;
  try {
    El::Environment env(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    const bool is_master = myid == 0;

    El::Grid grid(El::mpi::COMM_WORLD, 1, El::COLUMN_MAJOR);

    // TODO: load the data up from disk
    // TODO(later): use alchemist to load pre-processed data form spark
    // For compatibility with BLAS, local matrices are column major. Hence, we store
    // one instance per column and distribute the matrix by columns
    El::DistMatrix<> X(1, N, grid);

    // example where we sample N/2 points from a standard normal centered at +2 and the rest from one centered at -2
    El::Gaussian(X, 1, N);

    for (int j=0; j<X.LocalWidth(); ++j) {
      X.Matrix()(0, j) *= El::Sqrt(2);
      X.Matrix()(0, j) += j % 2 == 0 ? 1.0 : 0.0;
    }

    /* if (is_master) */
    /*   master_loop(); */
    /* else */
      worker_loop(myid, X);

  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
