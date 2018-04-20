#include "sampler.h"
#include "sgld_model.h"

namespace dsgld {

template <typename Field, typename T>
Sampler<Field, T>::Sampler(SGLDModel<Field, T>* model, const MPI_Comm& worker_comm)
    : model(model)
    , exchangeChains(true)
    , balanceLoads(true)
    , meanTrajectoryLength(1)
    , worker_comm(worker_comm)
    , A_(0.000001)
    , B_(1000.0)
    , C_(0.6)
{
  this->trajectory_length.resize(El::mpi::Size()-1);
  fill(trajectory_length.begin(), trajectory_length.end(), this->meanTrajectoryLength);
}

template <typename Field, typename T>
double Sampler<Field, T>::A() const {
  return this->A_;
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::A(const double A_) {
  this->A_= A_;
  return this;
}

template <typename Field, typename T>
double Sampler<Field, T>::B() const {
  return this->B_;
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::B(const double B_) {
  this->B_=B_;
  return this;
}
template <typename Field, typename T>
double Sampler<Field, T>::C() const {
  return this->C_;
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::C(const double C_) {
  this->C_= C_;
  return this;
}

template <typename Field, typename T>
int Sampler<Field, T>::MeanTrajectoryLength() const {
  return this->meanTrajectoryLength;
}

template <typename Field, typename T>
int Sampler<Field, T>::TrajectoryLength() const {
  return this->trajectory_length[El::mpi::Rank(worker_comm)];
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::MeanTrajectoryLength(const int meanTrajectoryLength) {
  this->meanTrajectoryLength = meanTrajectoryLength;
  fill(trajectory_length.begin(), trajectory_length.end(), this->meanTrajectoryLength);
  return this;
}

template <typename Field, typename T>
bool Sampler<Field, T>::ExchangeChains() const {
  return this->exchangeChains;
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::ExchangeChains(const bool exchangeChains) {
  this->exchangeChains = exchangeChains;
  return this;
}

template <typename Field, typename T>
bool Sampler<Field, T>::BalanceLoads() const {
  return this->balanceLoads;
}

template <typename Field, typename T>
Sampler<Field, T>* Sampler<Field, T>::BalanceLoads(const bool balanceLoads) {
  this->balanceLoads = balanceLoads;
  return this;
}

template <typename Field, typename T>
void Sampler<Field, T>::sampling_loop(
    const bool is_master,
    El::DistMatrix<Field>& thetaGlobal,
    const int n_samples) {
  const int n_traj = ceil(1.0 * n_samples / this->meanTrajectoryLength);

  // start with local copy
  El::Matrix<Field> theta0 = thetaGlobal.Matrix();

  El::Matrix<Field> theta = theta0;
  // TODO: fix this hack, uneven trajectory lengths means that these need to resize
  El::Matrix<Field> samples(model->d, 3*n_samples);
  El::Matrix<double> sampling_latencies(El::mpi::Size(), n_traj);
  El::Matrix<double> iteration_latencies(1, n_traj);
  vector<int> permutation(El::mpi::Size()-1);
  int t = 0;
  int samples_index = 0;
  int num_flushes = 0;
  for (int iter = 0; iter < n_traj; ++iter) {

    // sample a trajectory
    double iteration_start_time = MPI_Wtime();
    if (!is_master) {
      El::Output("Sampling trajectory: " + std::to_string(iter+1) + " out of " + std::to_string(n_traj));
      for (int traj_idx = 0; traj_idx < this->TrajectoryLength(); ++traj_idx) {

        samples(El::ALL, samples_index) = theta;

        // compute new step size
        double epsilon = this->A_ / El::Pow(1.0 + t / this->B_, this->C_);

        // perform update
        makeStep(epsilon, theta);

        t++;
        samples_index++;
      }
    }
    const double sampling_latency = MPI_Wtime() - iteration_start_time;

    // gather results and statistics from workers
    // first gather sampling_latencies so master can start scheduling next round while we update DistMatrix
    // TODO: avoid extra buffer alloc, e.g. MPI_Gather directly into the buffer
    double sampling_latencies_gather_buff[El::mpi::Size()];
    MPI_Gather(&sampling_latency, 1, MPI_DOUBLE, &sampling_latencies_gather_buff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (is_master) {
      sampling_latencies(El::ALL, iter) = El::Matrix<Field>(El::mpi::Size(), 1, &sampling_latencies_gather_buff[0], 1);
    }

    // use trajectory lengths to balance loads if needed
    if (this->balanceLoads) {
      rebalanceTrajectoryLengths(&sampling_latencies_gather_buff[0]);
    }

    // update distributed theta matrix on workers
    if (!is_master) {
      thetaGlobal.Reserve(theta.Height());
      for (int i=0; i<theta.Height(); ++i) {
        // TODO: bug in Elemental's RowShift? This is a column offset
        // TODO: bug in Elemental's QueueUpdate? need to subtract theta0 so this is actually an increment
        if (iter == 0) {
          thetaGlobal.QueueUpdate(i, thetaGlobal.RowShift(), theta(i) - theta0(i));
        } else {
          thetaGlobal.QueueUpdate(i, permutation[El::mpi::Rank(worker_comm)], theta(i) - theta0(i));
        }
      }
      thetaGlobal.ProcessQueues();
    }

    // schedule next round using a random permutation
    // QUESTION: why does this go wrong when wrapped in is_master?
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      permutation[i] = i;
    }
    if (this->exchangeChains) {
      std::random_shuffle(permutation.begin(), permutation.end());
    }
    //TODO: move one line up?
    MPI_Bcast(&permutation[0], El::mpi::Size()-1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!is_master) {
      const int next_theta_col_idx = permutation[El::mpi::Rank(worker_comm)];
      thetaGlobal.ReservePulls(theta.Height());
      for (int i=0; i<theta.Height(); ++i) {
        thetaGlobal.QueuePull(i, next_theta_col_idx);
      }
      thetaGlobal.ProcessPullQueue(theta0.Buffer());
    }
    theta = theta0;

    if (is_master) {
      iteration_latencies(0, iter) = MPI_Wtime() - iteration_start_time;
    }

    // write samples to disk
    El::Write(samples(El::ALL, El::IR(0,samples_index)), "samples-" + std::to_string(El::mpi::Rank()) + "-" + std::to_string(num_flushes), El::MATRIX_MARKET);
    num_flushes += 1;
    samples_index = 0;
  }
  if (is_master) {
    El::Write(sampling_latencies, "sampling_latencies", El::MATRIX_MARKET);
    El::Write(iteration_latencies, "iteration_latencies", El::MATRIX_MARKET);
  }
}

template <typename Field, typename T>
void Sampler<Field, T>::rebalanceTrajectoryLengths(double* sampling_latencies) {
  // TODO: this stops load balancing after first iteration to avoid oscillation
  // may want to periodically retoggle
  this->balanceLoads = false;

  bool is_master = El::mpi::Rank() == 0;
  if (is_master) {
    // update trajectory lengths for load balancing
    double sum_of_speeds = 0.0;
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      double speed = 1.0 / sampling_latencies[i+1];
      sum_of_speeds += speed;
    }
    for (int i=0; i<El::mpi::Size()-1; ++i) {
      double speed = 1.0 / sampling_latencies[i+1];
      // This results in drifting trajectory lengths
      /* trajectory_length[i] = ceil(speed * trajectory_length[i] / sum_of_speeds * (El::mpi::Size() - 1.0)); */

      // This results in alternating lengths if done every iteration
      trajectory_length[i] = ceil(speed / sum_of_speeds * this->meanTrajectoryLength * (El::mpi::Size() - 1.0));
    }
  }
  MPI_Bcast(&trajectory_length[0], El::mpi::Size()-1, MPI_INT, 0, MPI_COMM_WORLD);
}

template class Sampler<double, double>;
template class Sampler<double, int>;

} // namespace dsgld
