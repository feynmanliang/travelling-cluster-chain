#include "sgrld_sampler.h"
#include "sgld_model.h"

using std::vector;

namespace dsgld {

SGRLDSampler::SGRLDSampler(LDAModel* model)
    : Sampler<double, int>(model)
    , numGibbsSteps_(100)
{
}


int SGRLDSampler::NumGibbsSteps() const {
  return this->numGibbsSteps_;
}

SGRLDSampler* SGRLDSampler::NumGibbsSteps(const int numSteps) {
  this->numGibbsSteps_ = numSteps;
  return this;
}

void SGRLDSampler::makeStep(const double& epsilon, El::Matrix<double>& theta) {
  auto theta0 = theta; // make copy of original value

  // Gradient of log prior
  El::Axpy(double(epsilon / 2.0), this->model->nablaLogPrior(theta0), theta);

  // SGLD estimator
  El::Axpy(double(epsilon / 2.0 * this->model->N), this->model->sgldEstimate(theta0), theta);

  // Injected Gaussian noise
  El::Matrix<double> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());
  for (int i=0; i<theta.Height(); ++i) {
    for (int j=0; j<theta.Width(); ++j) {
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
