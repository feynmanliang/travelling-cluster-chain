#ifndef _DSGLD_GMM_TOY_MODEL_H__
#define _DSGLD_GMM_TOY_MODEL_H__


#include <El.hpp>

#include "sgld_model.h"

namespace dsgld {

class GMMToyModel : public SGLDModel<double, double> {
 public:
   GMMToyModel(const El::Matrix<double>& X, const int d);

  ~GMMToyModel() {};

  virtual El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) override;

  virtual El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const override;
};

}  // namespace dsgld


#endif  // _DSGLD_GMM_TOY_MODEL_H__

