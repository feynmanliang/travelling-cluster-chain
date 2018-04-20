#ifndef _DSGLD_LDA_MODEL_H__
#define _DSGLD_LDA_MODEL_H__

#include <El.hpp>

#include "sgld_model.h"

using std::string;
using std::vector;

namespace dsgld {

class LDAModel : public SGLDModel<double, int> {
 public:
   LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta);

  ~LDAModel() {};

  El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) override;

  El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const override;

  void writePerplexities(const string& filename);

  int NumGibbsSteps() const;
  LDAModel* NumGibbsSteps(const int);

  int BatchSize() const {
    return this->batchSize;
  }
  LDAModel* BatchSize(const int batchSize) {
    this->batchSize = batchSize;
    return this;
  }

 private:
  const double alpha_;
  const double beta_;
  int numGibbsSteps_;
  vector<double> perplexities_;
  int batchSize;
};

}  // namespace dsgld


#endif  // _DSGLD_LDA_MODEL_H__

