#include "mpi.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <list>
#include <stdio.h>
#include <iostream>

#include <El.hpp>

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::sort;
using std::string;
using std::normal_distribution;

void master_loop() {
  std::cout << " master" << std::endl;
}

template<typename Field>
void sample_traj(const int& c, El::Matrix<Field>& theta) {
  const double epsilon = 1.0; // step size
  const double N = 1.0; // dataset size

  normal_distribution<> d{0,epsilon};

  // TODO: nabla log prior

  // SGLD estimate
  El::Matrix<Field> sgldEstimate; // todo: make function of theta and subset X^n_t
  El::Gaussian(sgldEstimate, theta.Height(), theta.Width());

  El::Axpy(Field(epsilon / 2.0 * N), sgldEstimate, theta);

  // Injected Gaussian noise
  El::Matrix<Field> nu;
  El::Gaussian(nu, theta.Height(), theta.Width());

  El::Axpy(Field(1), nu, theta);
}

void worker_loop(const int& myid) {
  const int d = 5; // parameter dimension

  El::Matrix<double> theta;
  El::Ones(theta, d, 1);

  sample_traj(myid, theta);
}

int main(int argc, char** argv) {
  int myid, pnum;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum);

  const bool is_master = myid == 0;

  if (is_master)
    master_loop();
  else
    worker_loop(myid);

  MPI_Finalize();
  return 0;
}
