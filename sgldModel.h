#ifndef _DSGLD_MODEL_H__
#define _DSGLD_MODEL_H__

#include <El.hpp>

namespace dsgld {

// The SGLDModel class implements the problem-dependent functions required
// for performing SGLD sampling.
class SGLDModel {
 public:
   SGLDModel(const El::Matrix<double>& X, const int d)
     : N(X.Width()), d(d), X(X)
   {
   }

  virtual ~SGLDModel() { };

  // A SGLD gradient estimate computed for parameter settings theta and
  // minibatch X.
  virtual El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) const = 0;

  // Computes the gradient of the log prior distribution.
  virtual El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const = 0;

  const int N;
  const int d;

 protected:
  const El::Matrix<double>& X;
};

}  // namespace dsgld

#endif  // _DSGLD_MODEL_H__