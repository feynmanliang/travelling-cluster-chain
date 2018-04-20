#include "mpi.h"

#include <El.hpp>

#include "lda_model.h"
#include "sgld_model.h"
#include "sgrld_sampler.h"

using std::cout;
using std::endl;
using std::vector;

const double alpha = 0.1; // parameter to symmetric Dirichlet prior over topics
const double beta = 0.1; // parameter to symmetric Dirichlet prior over words
const int K = 10; // number of topics

const int N_SAMPLES = 100; // number of samples
const int TRAJ_LENGTH = 10; // trajectory length, number samples between exchanges, smaller => better mixing
const int N_GIBBS_STEPS = 5;

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

    El::DistMatrix<int> X(19889, 1740, grid);
    /* El::DistMatrix<double> thetaGlobal(K*W, El::mpi::Size(worker_comm), grid); */
    El::DistMatrix<double> thetaGlobal(grid);
    int N, W;
    if (!is_master) {
      // TODO(later): use alchemist to load pre-processed data form spark
      El::Read(X, "./test_data.mm.mtx", El::FileFormat::MATRIX_MARKET, false);
      N = X.Width();
      W = X.Height();

      // topic distributions, vectorized
      El::Ones(thetaGlobal, K*W, El::mpi::Size(worker_comm));
      // Prepare parameters
      // TODO: stick breaking initialization of the K topics
      thetaGlobal *= 1.0/W;
    }

    El::Matrix<int> X_local = X.Matrix();
    dsgld::LDAModel* model = new dsgld::LDAModel(X_local, K, alpha, beta);
    dsgld::Sampler<double, int>* sampler = (new dsgld::SGRLDSampler(model))
      ->NumGibbsSteps(N_GIBBS_STEPS)
      ->BalanceLoads(true)
      ->ExchangeChains(true);
    sampler->sampling_loop(worker_comm, is_master, thetaGlobal, N_SAMPLES, TRAJ_LENGTH);
    if (!is_master) {
      model->writePerplexities("perplexities-" + std::to_string(El::mpi::Rank()));
    }
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
