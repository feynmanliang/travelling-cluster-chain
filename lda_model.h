#ifndef _DSGLD_LDA_MODEL_H__
#define _DSGLD_LDA_MODEL_H__

#include <El.hpp>
#include <gsl/gsl_rng.h>

#include "sgld_model.h"

using std::string;
using std::vector;

namespace dsgld {

class LDAModel : public SGLDModel<double, int> {
 public:
   LDAModel(
       const El::Matrix<int>& X,
       const int K,
       const double alpha,
       const double beta);

  ~LDAModel() {};

  El::Matrix<double> sgldEstimate(const El::Matrix<double>& theta) override;

  El::Matrix<double> nablaLogPrior(const El::Matrix<double>& theta) const override;

  void writePerplexities(const string& filename);

  int NumGibbsSteps() const;
  LDAModel* NumGibbsSteps(const int);

 protected:
  void gibbsSample(
      const El::Matrix<int>& doc,
      const El::Matrix<double>& theta,
      El::Matrix<int>& index_to_topic,
      El::Matrix<int>& topic_counts) const;

  double estimatePerplexity(
      const El::Matrix<double>& theta,
      const El::Matrix<double>& theta_sum_over_w,
      const int num_words_in_doc,
      const El::Matrix<int>& topic_counts) const;

 private:
  const double alpha_;
  const double beta_;
  const int W;
  const int K;
  int numGibbsSteps_;
  vector<double> perplexities_;
  const gsl_rng* rng;
};

}  // namespace dsgld


#endif  // _DSGLD_LDA_MODEL_H__

