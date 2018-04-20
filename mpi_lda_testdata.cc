#include "mpi.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <El.hpp>

#include "lda_model.h"
#include "sgld_model.h"
#include "sgrld_sampler.h"

using std::cout;
using std::endl;
using std::vector;

const double alpha = 0.01; // parameter to symmetric Dirichlet prior over topics
const double beta = 0.01; // parameter to symmetric Dirichlet prior over words
const int K = 10; // number of topics

const int N_SAMPLES = 100; // number of samples

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
    const gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);

    const int N = 1740;
    const int W = 19889;
    El::DistMatrix<int> X(W, N, grid);
    El::DistMatrix<double> thetaGlobal(K*W, El::mpi::Size(worker_comm), grid);
    if (!is_master) {
      // TODO(later): use alchemist to load pre-processed data form spark
      El::Read(X, "./test_data.mtx", El::FileFormat::MATRIX_MARKET, false);

      // Prepare parameters by sampling Dirichlet
      El::Zeros(thetaGlobal, K*W, El::mpi::Size(worker_comm));

      vector<double> alpha_vec(W);
      std::fill(alpha_vec.begin(), alpha_vec.end(), alpha);
      thetaGlobal.Reserve(thetaGlobal.Height());
      for (int k=0; k<K; ++k) {
        double theta[W];
        gsl_ran_dirichlet(rng, W, alpha_vec.data(), theta);
        for (int w=0; w<W; ++w) {
          thetaGlobal.QueueUpdate(k*W + w, thetaGlobal.RowShift(), theta[w]);
        }
      }
      thetaGlobal.ProcessQueues();
    }

    El::Matrix<int> X_local = X.Matrix();
    dsgld::SGLDModel<double, int>* model = (new dsgld::LDAModel(X_local, K, alpha, beta))
      ->NumGibbsSteps(10)
      ->BatchSize(50);
    dsgld::Sampler<double, int>* sampler = (new dsgld::SGRLDSampler(model, worker_comm))
      ->BalanceLoads(true) // only beneficial when TRAJ_LENGTH > 1
      ->ExchangeChains(true)
      ->MeanTrajectoryLength(10)
      ->A(0.000001)
      ->B(1000.0)
      ->C(0.6);
    sampler->sampling_loop(is_master, thetaGlobal, N_SAMPLES);
    if (!is_master) {
      reinterpret_cast<dsgld::LDAModel*>(sampler)->writePerplexities("perplexities-" + std::to_string(El::mpi::Rank()));
    }
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
