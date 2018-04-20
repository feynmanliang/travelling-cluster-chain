#ifndef _DSGLD_SAMPLER_H__
#define _DSGLD_SAMPLER_H__

#include "sampler.h"
#include "sgld_model.h"

namespace dsgld {

template <typename Field, typename T>
class SGLDSampler : public Sampler<Field, T> {
 public:
  SGLDSampler(SGLDModel<Field, T>* model, const MPI_Comm& worker_comm);

  ~SGLDSampler() {}

  void makeStep(const Field& epsilon, El::Matrix<Field>& theta) override;

};

}  // namespace dsgld

#endif  // _DSGLD_SAMPLER_H__
