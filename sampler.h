#ifndef _DSGLD_SAMPLER_H__
#define _DSGLD_SAMPLER_H__

#include "sgld_model.h"

namespace dsgld {

class SGLDSampler {
 public:
  SGLDSampler(SGLDModel* model);

  ~SGLDSampler() {}

  void sampling_loop(
      const MPI_Comm& worker_comm,
      const bool is_master,
      El::DistMatrix<double>& thetaGlobal,
      const int n_samples,
      const int mean_traj_length);

  bool ExchangeChains() const;
  SGLDSampler& ExchangeChains(const bool);

  bool BalanceLoads() const;
  SGLDSampler& BalanceLoads(const bool);

 protected:
  void sgldUpdate(const double& epsilon, El::Matrix<double>& theta) ;

 private:
  const SGLDModel* model;
  bool exchangeChains; // Illustration only, should be true for proper mixing
  bool balanceLoads;
};

}  // namespace dsgld

#endif  // _DSGLD_SAMPLER_H__
