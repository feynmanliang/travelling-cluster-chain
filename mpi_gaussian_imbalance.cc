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
const int N_SAMPLES = 10000; // number of samples
const int TRAJ_LENGTH = N_SAMPLES / 5; // trajectory length, number samples between exchanges
const int N_TRAJ = ceil(1.0 * N_SAMPLES / TRAJ_LENGTH);
const double RANK_0_IMBALANCE = 0.2;

template <typename T>
T normal_pdf(T x, T m, T s) {
  static const T inv_sqrt_2pi = 0.3989422804014327;
  T a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}

template<typename Field, typename T>
El::Matrix<Field> sgldEstimate(const El::Matrix<Field>& theta, const El::Matrix<T>& X) {
  El::Matrix<Field> sgldEstimate(theta.Height(), theta.Width(), true);
  El::Zeros(sgldEstimate, theta.Height(), theta.Width());

  auto miniBatch = X;

  // TODO: remove
  /* miniBatch = miniBatch(El::ALL, El::IR(rand() % miniBatch.Width())); */

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

template<typename Field, typename T>
void sgldUpdate(const double& epsilon, El::Matrix<Field>& theta, const El::Matrix<T>& X) {
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

template<typename Field, typename T>
void sampling_loop(const MPI_Comm& worker_comm, const bool is_master, El::DistMatrix<Field>& thetaGlobal, const El::Matrix<T>& X) {
  // start with local copy
  El::Matrix<T> theta0 = thetaGlobal.Matrix();


  El::Matrix<T> theta = theta0;
  // TODO: fix this hack, uneven trajectory lengths means that these need to resize
  El::Matrix<T> samples(d, 3*N_SAMPLES);
  El::Matrix<T> sampling_latencies(El::mpi::Size(), N_TRAJ);
  El::Matrix<T> iteration_latencies(1, N_TRAJ);
  vector<double> permutation(El::mpi::Size()-1);
  vector<int> trajectory_length(El::mpi::Size()-1);
  fill(trajectory_length.begin(), trajectory_length.end(), TRAJ_LENGTH);
  int t = 0;
  for (int iter = 0; iter < N_TRAJ; ++iter) {

    // sample a trajectory
    double iteration_start_time = MPI_Wtime();
    if (!is_master) {
      for (int traj_idx = 0; traj_idx < trajectory_length[El::mpi::Rank(worker_comm)]; ++traj_idx) {

        /* if (t % 100 == 0 && is_master) */
        /*   El::Output("Sampling " + std::to_string(t) + "/" + std::to_string(N_SAMPLES)); */

        samples(El::ALL, t) = theta;

        // compute new step size
        double epsilon = 0.04 / El::Pow(10.0 + t, 0.55);

        // perform sgld update
        sgldUpdate(epsilon, theta, X);

        t++;
      }
    }
    const double sampling_latency = MPI_Wtime() - iteration_start_time;

    // gather results and statistics from workers
    // first gather sampling_latencies so master can start scheduling next round while we update DistMatrix
    double sampling_latencies_gather_buff[El::mpi::Size()];
    MPI_Gather(&sampling_latency, 1, MPI_DOUBLE, &sampling_latencies_gather_buff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (is_master) {
      sampling_latencies(El::ALL, iter) = std::move(El::Matrix<double>(El::mpi::Size(), 1, &sampling_latencies_gather_buff[0], 1));
    }

    // update distributed theta matrix on workers
    if (!is_master) {
      thetaGlobal.Reserve(theta.Height());
      for (int i=0; i<theta.Height(); ++i) {
        // TODO: bug in Elemental's RowShift? This is a column offset
        // TODO: bug in Elemental's QueueUpdate? need to subtract theta0 so this is actually an increment
        if (iter == 0) {
          thetaGlobal.QueueUpdate(i, thetaGlobal.RowShift(), theta(i) - theta0(i));
        } else {
          thetaGlobal.QueueUpdate(i, permutation[El::mpi::Rank(worker_comm)], theta(i) - theta0(i));
        }
      }
      thetaGlobal.ProcessQueues();
    }

    // schedule next round using a random permutation
    // TODO: use latency information instead
    // QUESTION: why does this go wrong when wrapped in is_master?
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      permutation[i] = i;
    }
    // NOTE: comment the following line to not exchange chains
    std::random_shuffle(permutation.begin(), permutation.end());
    MPI_Bcast(&permutation[0], El::mpi::Size()-1, MPI_INT, 0, MPI_COMM_WORLD);

    // update trajectory lengths for load balancing
    double sum_of_speeds = 0.0;
    if (is_master) {
      El::Print(sampling_latencies(El::ALL, iter));
    }
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      double speed = 1.0 / sampling_latencies(i+1, iter);
      sum_of_speeds += speed;
      trajectory_length[i] = TRAJ_LENGTH * speed;
    }
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      trajectory_length[i] = ceil(trajectory_length[i] / sum_of_speeds);
    }
    MPI_Bcast(&trajectory_length[0], El::mpi::Size()-1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!is_master) {
      const int next_theta_col_idx = permutation[El::mpi::Rank(worker_comm)];
      thetaGlobal.ReservePulls(theta.Height());
      for (int i=0; i<theta.Height(); ++i) {
        thetaGlobal.QueuePull(i, next_theta_col_idx);
      }
      thetaGlobal.ProcessPullQueue(theta0.Buffer());
    }
    theta = theta0;

    if (is_master) {
      iteration_latencies(0, iter) = MPI_Wtime() - iteration_start_time;
    }
  }
  // write samples to disk
  El::Write(samples, "samples-" + std::to_string(El::mpi::Rank()), El::MATRIX_MARKET);
  if (is_master) {
    El::Write(sampling_latencies, "sampling_latencies", El::MATRIX_MARKET);
    El::Write(iteration_latencies, "iteration_latencies", El::MATRIX_MARKET);
  }
}


int main(int argc, char** argv) {
  try {
    El::Environment env(argc, argv);
    const int world_rank = El::mpi::Rank();
    const bool is_master = world_rank == 0;

    // Worker comm is used by Elemental to ensure matrices only partitioned across workers
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_master, world_rank, &worker_comm);
    El::Grid grid(worker_comm, 1, El::COLUMN_MAJOR);

    srand(42 + El::mpi::Rank());

    const int N_local = El::mpi::Rank(worker_comm) == 0
      ? floor(RANK_0_IMBALANCE * N)
      : ceil((1.0 - RANK_0_IMBALANCE) * N / (El::mpi::Size(worker_comm) - 1.0));
    El::Matrix<double> X(1, N_local);
    El::DistMatrix<double> thetaGlobal(d, El::mpi::Size(worker_comm), grid);
    if (!is_master) {
      // Prepare parameters
      El::Ones(thetaGlobal, d, El::mpi::Size(worker_comm));

      // Prepare data
      // TODO(later): use alchemist to load pre-processed data form spark
      // For compatibility with BLAS, local matrices are column major. Hence, we store
      // one instance per column and distribute the matrix by columns

      // example where we sample N/2 points from a standard normal centered at +2 and the rest from one centered at -2
      El::Gaussian(X, 1, N_local);
      X *= El::Sqrt(2.0); // variance 2

      for (int j=0; j<X.Width(); ++j) {
        X(0, j) += (rand() % 2 == 0 ? 1.0 : 0.0);
      }
    }
    sampling_loop(worker_comm, is_master, thetaGlobal, X);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
