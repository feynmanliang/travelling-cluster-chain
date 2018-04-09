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

using std::cout;
using std::endl;
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
    sgldEstimate(0, 0) += score0 * (x - theta(0)) / 2.0;
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

struct SampleTrajResponse {
  double latency;
  double firstValue;
};
MPI_Datatype SampleTrajResponseType;

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
void sampling_loop(MPI_Comm worker_comm, const El::DistMatrix<T>& X) {
  const bool is_master = El::mpi::Rank() == 0;

  // distributed initialization on workers
  El::Matrix<double> theta(d, 1);
  if (!is_master) {
    El::Ones(theta, d, 1);
  }

    // TODO?: burn in before collecting samples

  El::Matrix<double> samples(d, N_SAMPLES);
  for (int i = 0; i < N_SAMPLES; ++i) {
    double t_start, t_end;
    if (!is_master) {
      t_start = MPI_Wtime();

      //TODO: trajectory sampling

      // save a sample
      samples(El::ALL, i) = theta;

      // compute new step size
      double epsilon = 0.04 / El::Pow(10.0 + i, 0.55);

      // perform sgld update
      sgldUpdate(epsilon, theta, X);
      t_end = MPI_Wtime();
    }
    SampleTrajResponse resp = {
      t_end - t_start,
      theta(0)
    };
    // TODO: gather results and statistics from workers
    SampleTrajResponse recvbuff[El::mpi::Size(worker_comm)];
    MPI_Gather(&resp, 1, SampleTrajResponseType, &recvbuff, 1, SampleTrajResponseType, 0, MPI_COMM_WORLD);
    if (is_master) {
      cout << "Finished gather: " << endl;
      cout << recvbuff[1].latency << endl;
      cout << recvbuff[1].firstValue << endl;
    }
    // TODO: schedule next round
    // TODO: send off next round

    // write all samples to disk
    El::Write(samples, "samples-" + std::to_string(El::mpi::Rank()), El::MATRIX_MARKET);
  }
}


int main(int argc, char** argv) {
  try {
    El::Environment env(argc, argv);
    const int world_rank = El::mpi::Rank();
    const int world_size = El::mpi::Size();
    const bool is_master = world_rank == 0;

    // Worker comm is used by Elemental to ensure matrices only partitioned across workers
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_master, world_rank, &worker_comm);
    El::Grid grid(worker_comm, 1, El::COLUMN_MAJOR);

    // declare type
    MPI_Datatype type[2] = { MPI_DOUBLE };
    int blocklen[1] = {2};
    MPI_Aint disp[1] = {0};
    MPI_Type_create_struct(1, blocklen, disp, type, &SampleTrajResponseType);
    MPI_Type_commit(&SampleTrajResponseType);

    int row_rank, row_size;
    MPI_Comm_rank(worker_comm, &row_rank);
    MPI_Comm_size(worker_comm, &row_size);

    El::DistMatrix<> X(1, N, grid);
    if (!is_master) {
      // Prepare data
      // TODO(later): use alchemist to load pre-processed data form spark
      // For compatibility with BLAS, local matrices are column major. Hence, we store
      // one instance per column and distribute the matrix by columns

      // example where we sample N/2 points from a standard normal centered at +2 and the rest from one centered at -2
      El::Gaussian(X, 1, N);
      X *= El::Sqrt(2.0); // variance 2

      for (int j=0; j<X.LocalWidth(); ++j) {
        X.Matrix()(0, j) += (j % 2 == 0 ? 1.0 : 0.0);
      }
    }
    sampling_loop(worker_comm, X);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
