#include "sgrld_sampler.h"
#include "sgld_model.h"

using std::vector;

namespace dsgld {

SGRLDSampler::SGRLDSampler(LDAModel* model)
    : Sampler<double, int>(model)
{
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
  El::Axpy(El::Sqrt(epsilon), nu, theta);
}

} // namespace dsgld
