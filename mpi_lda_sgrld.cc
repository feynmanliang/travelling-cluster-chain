#include "mpi.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <El.hpp>

#include "lda_model.h"
#include "sgrld_sampler.h"

using std::vector;
using std::cout;
using std::endl;

const double alpha = 0.01; // parameter to symmetric Dirichlet prior over topics
const double beta = 0.01; // parameter to symmetric Dirichlet prior over words
const int K = 3; // number of topics
const int N = 5; // number of documents, NOTE: per worker here
const int W = 20; // number of words (vocab size)

const int N_SAMPLES = 100;

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

    El::Matrix<int> X_local(W, N);
    // topic distributions, vectorized
    El::DistMatrix<double> thetaGlobal(K*W, El::mpi::Size(worker_comm), grid);
    El::Zeros(thetaGlobal, K*W, El::mpi::Size(worker_comm));
    if (!is_master) {
      // Prepare parameters by sampling Dirichlet
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

      // Prepare data
      // TODO(later): use alchemist to load pre-processed data form spark
      for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
          X_local(i,j) = rand() % 300;
        }
      }
    }

    dsgld::SGLDModel<double, int>* model = (new dsgld::LDAModel(X_local, K, alpha, beta))
      ->NumGibbsSteps(10)
      ->BatchSize(N);
    dsgld::Sampler<double, int>* sampler = (new dsgld::SGRLDSampler(N, model, worker_comm))
      ->BalanceLoads(true) // only beneficial when TRAJ_LENGTH > 1
      ->ExchangeChains(true)
      ->MeanTrajectoryLength(N_SAMPLES / 10)
      ->A(1e-4)
      ->B(100.0)
      ->C(0.6);
    sampler->sampling_loop(is_master, thetaGlobal, N_SAMPLES);
    reinterpret_cast<dsgld::LDAModel*>(model)->writePerplexities("perplexities-" + std::to_string(El::mpi::Rank()));
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
