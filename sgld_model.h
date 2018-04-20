#ifndef _DSGLD_MODEL_H__
#define _DSGLD_MODEL_H__

#include <El.hpp>

namespace dsgld {

// The SGLDModel class implements the problem-dependent functions required
// for performing SGLD sampling.
template <typename Field, typename T>
class SGLDModel {
 public:
   // TODO: take theta directly instead of d?
   SGLDModel(const El::Matrix<T>& X, const int d)
     : N(X.Width())
     , d(d)
     , X(X)
     , minibatchIter(0)
   {
   }

  virtual ~SGLDModel() { };

  // A SGLD gradient estimate computed for parameter settings theta and
  // minibatch X.
  virtual El::Matrix<Field> sgldEstimate(const El::Matrix<Field>& theta) = 0;

  // Computes the gradient of the log prior distribution.
  virtual El::Matrix<Field> nablaLogPrior(const El::Matrix<Field>& theta) const = 0;

  const int N;
  const int d;

 protected:
  const El::Matrix<T>& X;
  int minibatchIter;
};

}  // namespace dsgld

#endif  // _DSGLD_MODEL_H__
