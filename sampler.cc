#include "sampler.h"
#include "sgld_model.h"

using std::vector;

namespace dsgld {

template <typename Field>
SGLDSampler<Field>::SGLDSampler(SGLDModel* model)
    : model(model), exchangeChains(true), balanceLoads(true)
{
}

template <typename Field>
bool SGLDSampler<Field>::ExchangeChains() const {
  return this->exchangeChains;
}

template <typename Field>
SGLDSampler<Field>& SGLDSampler<Field>::ExchangeChains(const bool exchangeChains) {
  this->exchangeChains = exchangeChains;
  return *this;
}

template <typename Field>
bool SGLDSampler<Field>::BalanceLoads() const {
  return this->balanceLoads;
}

template <typename Field>
SGLDSampler<Field>& SGLDSampler<Field>::BalanceLoads(const bool balanceLoads) {
  this->balanceLoads = balanceLoads;
  return *this;
}

template <typename Field>
void SGLDSampler<Field>::sgldUpdate(const Field& epsilon, El::Matrix<Field>& theta) {
  auto theta0 = theta; // make copy of original value

  // Gradient of log prior
  El::Axpy(Field(epsilon / 2.0), model->nablaLogPrior(theta0), theta);

  // SGLD estimator
  El::Axpy(Field(epsilon / 2.0 * model->N), model->sgldEstimate(theta0), theta);

  // Injected Gaussian noise
  El::Matrix<Field> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());
  El::Axpy(El::Sqrt(epsilon), nu, theta);
}

template <typename Field>
void SGLDSampler<Field>::sampling_loop(
    const MPI_Comm& worker_comm,
    const bool is_master,
    El::DistMatrix<Field>& thetaGlobal,
    const int n_samples,
    const int mean_traj_length) {
  // TODO: work in
  const int n_traj = ceil(1.0 * n_samples / mean_traj_length);

  // start with local copy
  El::Matrix<Field> theta0 = thetaGlobal.Matrix();

  El::Matrix<Field> theta = theta0;
  // TODO: fix this hack, uneven trajectory lengths means that these need to resize
  El::Matrix<Field> samples(model->d, 3*n_samples);
  El::Matrix<Field> sampling_latencies(El::mpi::Size(), n_traj);
  El::Matrix<Field> iteration_latencies(1, n_traj);
  vector<int> permutation(El::mpi::Size()-1);
  vector<int> trajectory_length(El::mpi::Size()-1);
  fill(trajectory_length.begin(), trajectory_length.end(), mean_traj_length);
  int t = 0;
  for (int iter = 0; iter < n_traj; ++iter) {

    // sample a trajectory
    double iteration_start_time = MPI_Wtime();
    if (!is_master) {
      for (int traj_idx = 0; traj_idx < trajectory_length[El::mpi::Rank(worker_comm)]; ++traj_idx) {

        /* if (t % 100 == 0 && is_master) */
        /*   El::Output("Sampling " + std::to_string(t) + "/" + std::to_string(n_samples)); */

        samples(El::ALL, t) = theta;

        // compute new step size
        double epsilon = 0.04 / El::Pow(10.0 + t, 0.55);

        // perform sgld update
        sgldUpdate(epsilon, theta);

        t++;
      }
    }
    const double sampling_latency = MPI_Wtime() - iteration_start_time;

    // gather results and statistics from workers
    // first gather sampling_latencies so master can start scheduling next round while we update DistMatrix
    double sampling_latencies_gather_buff[El::mpi::Size()];
    MPI_Gather(&sampling_latency, 1, MPI_DOUBLE, &sampling_latencies_gather_buff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (is_master) {
      // TODO: avoid caling constructor, maybe std::move, or if could MPI_Gather directly into the buffer
      sampling_latencies(El::ALL, iter) = El::Matrix<Field>(El::mpi::Size(), 1, &sampling_latencies_gather_buff[0], 1);

      // update trajectory lengths for load balancing
      double sum_of_speeds = 0.0;
      for (int i=0; i<El::mpi::Size()-1; ++i) {
        double speed = 1.0 / sampling_latencies(i+1, iter);
        sum_of_speeds += speed;
      }
      for (int i=0; i<El::mpi::Size()-1; ++i) {
        double speed = 1.0 / sampling_latencies(i+1, iter);
        trajectory_length[i] = ceil(speed * trajectory_length[i] / sum_of_speeds * (El::mpi::Size() - 1.0));

        if (!this->balanceLoads) {
          trajectory_length[i] = mean_traj_length;
        }
      }
    }
    MPI_Bcast(&trajectory_length[0], El::mpi::Size()-1, MPI_INT, 0, MPI_COMM_WORLD);

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
  }
  El::mpi::Barrier();

  // write samples to disk
  El::Write(samples(El::ALL, El::IR(0,t)), "samples-" + std::to_string(El::mpi::Rank()), El::MATRIX_MARKET);
  if (is_master) {
    El::Write(sampling_latencies, "sampling_latencies", El::MATRIX_MARKET);
    El::Write(iteration_latencies, "iteration_latencies", El::MATRIX_MARKET);
  }
}

template class SGLDSampler<double>;

} // namespace dsgld
