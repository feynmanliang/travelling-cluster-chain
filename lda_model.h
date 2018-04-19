#ifndef _DSGLD_LDA_MODEL_H__
#define _DSGLD_LDA_MODEL_H__

#include <El.hpp>

#include "sgld_model.h"

namespace dsgld {

class LDAModel : public SGLDModel<double, int> {
 public:
   LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta);

  ~LDAModel() {};

  El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) const override;

  El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const override;

 private:
  const double alpha_;
  const double beta_;
};

}  // namespace dsgld


#endif  // _DSGLD_LDA_MODEL_H__

