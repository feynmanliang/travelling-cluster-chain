#include "sgrld_sampler.h"
#include "sgld_model.h"

using std::vector;

namespace dsgld {

SGRLDSampler::SGRLDSampler(const int N, SGLDModel<double, int>* model, MPI_Comm& worker_comm)
    : Sampler<double, int>(N, model, worker_comm)
{
}

void SGRLDSampler::makeStep(const double& epsilon, El::Matrix<double>& theta) {
  auto theta0 = theta; // make copy of original value

  // Gradient of log prior
  El::Axpy(double(epsilon / 2.0), this->model->nablaLogPrior(theta0), theta);

  // SGLD estimator
  const double q = 1.0 * this->TrajectoryLength() / (this->MeanTrajectoryLength() * (El::mpi::Size()-1));
  El::Axpy(double((epsilon / 2.0) * this->model->N / (this->N_total * q)), this->model->sgldEstimate(theta0), theta);

  // Injected Gaussian noise
  El::Matrix<double> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());
  for (int i=0; i<theta.Height(); ++i) {
    for (int j=0; j<theta.Width(); ++j) {
      // Preconditioning by Riemannian metric tensor
      nu(i,j) *= El::Sqrt(theta0(i,j));
    }
  }
  El::Axpy(El::Sqrt(epsilon), nu, theta);

  // Mirroring to stay on probability simplex
  for (int i=0; i<theta.Height(); ++i) {
    for (int j=0; j<theta.Width(); ++j) {
      theta(i,j) = El::SafeAbs(theta(i,j));
    }
  }
}

} // namespace dsgld
