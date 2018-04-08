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
#include <random>

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

void sample_traj(const int& c, vector<double>& theta) {
  const double epsilon = 1.0;

  std::random_device rd{};
  std::mt19937 gen{rd()};

  normal_distribution<> d{0,epsilon};

  std::cout << theta[0] << " " << theta[1] << std::endl;
  for (auto &t : theta) {
    t += (epsilon / 2.0) * d(gen) + d(gen);
  }
  std::cout << theta[0] << " " << theta[1] << std::endl;
}

void worker_loop(const int& myid) {
  vector<double> theta { 1.0, 1.0 };
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
