#ifndef _DSGLD_SAMPLER_H__
#define _DSGLD_SAMPLER_H__

#include "sgld_model.h"

namespace dsgld {

template <typename Field, typename T>
class SGLDSampler {
 public:
  SGLDSampler(SGLDModel<Field, T>* model);

  ~SGLDSampler() {}

  void sampling_loop(
      const MPI_Comm& worker_comm,
      const bool is_master,
      El::DistMatrix<Field>& thetaGlobal,
      const int n_samples,
      const int mean_traj_length);

  bool ExchangeChains() const;
  SGLDSampler& ExchangeChains(const bool);

  bool BalanceLoads() const;
  SGLDSampler& BalanceLoads(const bool);

 protected:
  void sgldUpdate(const Field& epsilon, El::Matrix<Field>& theta) ;

 private:
  const SGLDModel<Field, T>* model;
  bool exchangeChains; // Illustration only, should be true for proper mixing
  bool balanceLoads;
};

}  // namespace dsgld

#endif  // _DSGLD_SAMPLER_H__
