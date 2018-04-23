#include "sgld_sampler.h"
#include "sgld_model.h"

using std::vector;

namespace dsgld {

template <typename Field, typename T>
SGLDSampler<Field, T>::SGLDSampler(const int N, SGLDModel<Field, T>* model, const MPI_Comm& worker_comm)
    : Sampler<Field, T>(N, model, worker_comm)
{
}

template <typename Field, typename T>
void SGLDSampler<Field, T>::makeStep(const Field& epsilon, El::Matrix<Field>& theta) {
  auto theta0 = theta; // make copy of original value

  // Gradient of log prior
  El::Axpy(Field(epsilon / 2.0), this->model->nablaLogPrior(theta0), theta);

  // SGLD estimator, adjusting for bias introduced by unequal trajectory lengths
  const double q = 1.0 * this->TrajectoryLength() / (this->MeanTrajectoryLength() * (El::mpi::Size()-1));
  El::Axpy(Field((epsilon / 2.0) * this->N_total / q), this->model->sgldEstimate(theta0), theta);

  // Injected Gaussian noise
  El::Matrix<Field> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());
  El::Axpy(El::Sqrt(epsilon), nu, theta);
}

template class SGLDSampler<double, double>;
template class SGLDSampler<double, int>;

} // namespace dsgld
