#include "mpi.h"

#include <El.hpp>

#include "sampler.h"
#include "sgldModel.h"

using std::vector;
using std::cout;
using std::endl;

const int N = 100; // dataset size
const int d = 2; // parameter dimension
const int N_SAMPLES = 50000; // number of samples
const int TRAJ_LENGTH = N_SAMPLES / 5; // trajectory length, number samples between exchanges, smaller => better mixing
/* const int N_TRAJ = ceil(1.0 * N_SAMPLES / TRAJ_LENGTH); */
const double RANK_0_IMBALANCE = 0.95;

class GaussianImbalanceModel : public dsgld::SGLDModel {
 public:
   GaussianImbalanceModel(const El::Matrix<double>& X, const int d)
       : dsgld::SGLDModel(X, d)
   {
   }

  El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) const override {
      El::Matrix<double> sgldEstimate(theta.Height(), theta.Width(), true);
      El::Zeros(sgldEstimate, theta.Height(), theta.Width());

      auto miniBatch = this->X;

      // NOTE: uncomment to take single random sample as minibatch
      // leave commented to make cost of imbalance obvious
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

  // Computes the gradient of the log prior distribution.
  El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const override {
    El::Matrix<double> nablaLogPrior;

    // Gaussian prior centered at origin, should be -\theta / \sigma^2
    nablaLogPrior = theta;
    nablaLogPrior *= -1.0;
    nablaLogPrior(El::IR(0), 0) *= 0.1; // prior variance \sigma_1^2 = 10.0

    return nablaLogPrior;
  }


  // TODO: move to utils
 private:
   double normal_pdf(double x, double m, double s) const {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5 * a * a);
}
};

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
    cout << El::mpi::Rank(worker_comm) << " " << N_local << endl;
    GaussianImbalanceModel* model = new GaussianImbalanceModel(X, d);
    dsgld::SGLDSampler* sampler = new dsgld::SGLDSampler(model);
    sampler->sampling_loop(worker_comm, is_master, thetaGlobal, N_SAMPLES, TRAJ_LENGTH);
  } catch (std::exception& e) {
    El::ReportException(e);
    return 1;
  }
  return 0;
}
