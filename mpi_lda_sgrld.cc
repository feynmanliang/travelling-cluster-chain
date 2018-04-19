#include "mpi.h"

#include <El.hpp>

#include "lda_model.h"
#include "sampler.h"

using std::vector;
using std::cout;
using std::endl;

const double alpha = 0.1; // parameter to symmetric Dirichlet prior over topics
const double beta = 0.1; // parameter to symmetric Dirichlet prior over words
const int K = 10; // number of topics
const int N = 100; // number of documents, NOTE: per worker here
const int W = 100; // number of words (vocab size)

const int N_SAMPLES = 50000; // number of samples
const int TRAJ_LENGTH = N_SAMPLES / 5; // trajectory length, number samples between exchanges, smaller => better mixing

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

    El::Matrix<int> X_local(W, N);
    // topic distributions, vectorized
    El::DistMatrix<double> thetaGlobal(K*W, El::mpi::Size(worker_comm), grid);
    if (!is_master) {
      // Prepare parameters
      // TODO: stick breaking initialization of the K topics
      El::Ones(thetaGlobal, K*W, El::mpi::Size(worker_comm));

      // Prepare data
      // TODO(later): use alchemist to load pre-processed data form spark
      for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
          X_local(i,j) = rand() % 1000;
        }
      }
    }

    dsgld::LDAModel* model = new dsgld::LDAModel(X_local, K, alpha, beta);
    dsgld::SGLDSampler<double, int> sampler = (*(new dsgld::SGLDSampler<double, int>(model)))
      .BalanceLoads(true)
      .ExchangeChains(true);
    sampler.sampling_loop(worker_comm, is_master, thetaGlobal, N_SAMPLES, TRAJ_LENGTH);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
