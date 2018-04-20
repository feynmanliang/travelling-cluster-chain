#include "mpi.h"

#include <El.hpp>

#include "gmm_toy_model.h"
#include "sgld_sampler.h"

using std::vector;
using std::cout;
using std::endl;

const int N = 100; // dataset size
const int d = 2; // parameter dimension
const int N_SAMPLES = 50000; // number of samples
const double RANK_0_IMBALANCE = 0.95;

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
    dsgld::GMMToyModel* model = new dsgld::GMMToyModel(X, d);
    dsgld::Sampler<double, double>* sampler = (new dsgld::SGLDSampler<double, double>(model))
      ->BalanceLoads(true)
      ->ExchangeChains(true)
      ->MeanTrajectoryLength(N_SAMPLES/5);
    sampler->sampling_loop(worker_comm, is_master, thetaGlobal, N_SAMPLES);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
