#ifndef _SAMPLER_H__
#define _SAMPLER_H__

#include "sgld_model.h"

namespace dsgld {

template <typename Field, typename T>
class Sampler {
 public:
  Sampler(SGLDModel<Field, T>* model);

  ~Sampler() {}

  void sampling_loop(
      const MPI_Comm& worker_comm,
      const bool is_master,
      El::DistMatrix<Field>& thetaGlobal,
      const int n_samples,
      const int mean_traj_length);

  bool ExchangeChains() const;
  Sampler* ExchangeChains(const bool);

  bool BalanceLoads() const;
  Sampler* BalanceLoads(const bool);

 protected:
  const SGLDModel<Field, T>* model;
  virtual void makeStep(const Field& epsilon, El::Matrix<Field>& theta) = 0;

 private:
  bool exchangeChains; // Illustration only, should be true for proper mixing
  bool balanceLoads;
};

}  // namespace dsgld

#endif  // _SAMPLER_H__
