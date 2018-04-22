#ifndef _SAMPLER_H__
#define _SAMPLER_H__

#include "sgld_model.h"

using std::vector;

namespace dsgld {

template <typename Field, typename T>
class Sampler {
 public:
  Sampler(const int N_total, SGLDModel<Field, T>* model, const MPI_Comm& worker_comm);

  ~Sampler() {}

  void sampling_loop(
      const bool is_master,
      El::DistMatrix<Field>& thetaGlobal,
      const int n_samples);

  bool ExchangeChains() const;
  Sampler* ExchangeChains(const bool);

  bool BalanceLoads() const;
  Sampler* BalanceLoads(const bool);

  int MeanTrajectoryLength() const;
  Sampler* MeanTrajectoryLength(const int);

  int TrajectoryLength() const;

  double A() const;
  Sampler* A(const double);

  double B() const;
  Sampler* B(const double);

  double C() const;
  Sampler* C(const double);

 protected:
  SGLDModel<Field, T>* model;
  virtual void makeStep(const Field& epsilon, El::Matrix<Field>& theta) = 0;
  void rebalanceTrajectoryLengths(double* sampling_latencies);
  vector<int> trajectory_length;
  const int N_total;

 private:
  bool exchangeChains; // Illustration only, should be true for proper mixing
  bool balanceLoads;
  int meanTrajectoryLength;
  const MPI_Comm& worker_comm;
  double A_;
  double B_;
  double C_;
};

}  // namespace dsgld

#endif  // _SAMPLER_H__
