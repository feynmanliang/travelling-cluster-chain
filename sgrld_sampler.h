#ifndef _DSGLD_SGRLD_SAMPLER_H__
#define _DSGLD_SGRLD_SAMPLER_H__

#include "sampler.h"
#include "lda_model.h"

namespace dsgld {

class SGRLDSampler : public Sampler<double, int> {
 public:
  SGRLDSampler(LDAModel* model);

  ~SGRLDSampler() {}

  void makeStep(const double& epsilon, El::Matrix<double>& theta) override;

};

}  // namespace dsgld

#endif  // _DSGLD_SGRLD_SAMPLER_H__
