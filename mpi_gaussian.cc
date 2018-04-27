#include "mpi.h"

#include <El.hpp>

#include "gmm_toy_model.h"
#include "sgld_sampler.h"

using std::vector;
using std::cout;
using std::endl;

const int N = 100; // dataset size
const int d = 2; // parameter dimension
const int N_SAMPLES = 10000*N; // number of samples

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

    El::DistMatrix<double> X(1, N, grid);
    El::DistMatrix<double> thetaGlobal(d, El::mpi::Size(worker_comm), grid);
    if (!is_master) {
      // Prepare parameters
      El::Ones(thetaGlobal, d, El::mpi::Size(worker_comm));

      // Prepare data
      // example where we sample N/2 points from a standard normal centered at +2 and the rest from one centered at -2
      El::Gaussian(X, 1, N);
      X *= El::Sqrt(2.0); // variance 2

      for (int j=0; j<X.LocalWidth(); ++j) {
        X.Matrix()(0, j) += (rand() % 2 == 0 ? 1.0 : 0.0);
      }
    }
    El::Matrix<double> X_local = X.Matrix();

    dsgld::SGLDModel<double, double>* model = (new dsgld::GMMToyModel(X_local, d))
      ->BatchSize(1);
    dsgld::Sampler<double, double>* sampler =
      (new dsgld::SGLDSampler<double, double>(N, model, worker_comm))
      ->BalanceLoads(false)
      ->ExchangeChains(true)
      ->MeanTrajectoryLength(N_SAMPLES / 500)
      ->A(0.001)
      ->B(1000)
      ->C(0.55);
    sampler->sampling_loop(is_master, thetaGlobal, N_SAMPLES);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
