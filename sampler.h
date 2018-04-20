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
      const int n_samples);

  bool ExchangeChains() const;
  Sampler* ExchangeChains(const bool);

  bool BalanceLoads() const;
  Sampler* BalanceLoads(const bool);

  int MeanTrajectoryLength() const;
  Sampler* MeanTrajectoryLength(const int);

 protected:
  SGLDModel<Field, T>* model;
  void rebalanceTrajectoryLengths(double* sampling_latencies, int* trajectory_length);
  virtual void makeStep(const Field& epsilon, El::Matrix<Field>& theta) = 0;

 private:
  bool exchangeChains; // Illustration only, should be true for proper mixing
  bool balanceLoads;
  int meanTrajectoryLength;
};

}  // namespace dsgld

#endif  // _SAMPLER_H__
