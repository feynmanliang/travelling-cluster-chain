#include "gmm_toy_model.h"

namespace dsgld {

// TODO: move to utils
double normal_pdf(double x, double m, double s) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5 * a * a);
}

GMMToyModel::GMMToyModel(const El::Matrix<double>& X, const int d)
    : SGLDModel(X, d)
{
}

El::Matrix<double> GMMToyModel::sgldEstimate(const El::Matrix<double>& theta) const {
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

El::Matrix<double> GMMToyModel::nablaLogPrior(const El::Matrix<double>& theta) const {
    El::Matrix<double> nablaLogPrior;

    // Gaussian prior centered at origin, should be -\theta / \sigma^2
    nablaLogPrior = theta;
    nablaLogPrior *= -1.0;
    nablaLogPrior(El::IR(0), 0) *= 0.1; // prior variance \sigma_1^2 = 10.0

    return nablaLogPrior;
}


} // namespace dsgld
