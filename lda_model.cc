#include <random>

#include <gsl/gsl_randist.h>

#include "lda_model.h"

using std::vector;

namespace dsgld {

LDAModel::LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta)
    : SGLDModel<double, int>(X, K*X.Height())
    , alpha_(alpha)
    , beta_(beta)
    , W(X.Height())
    , K(K)
    , numGibbsSteps_(100)
    , rng(gsl_rng_alloc(gsl_rng_taus))
{
}

int LDAModel::NumGibbsSteps() const {
  return this->numGibbsSteps_;
}

LDAModel* LDAModel::NumGibbsSteps(const int numSteps) {
  this->numGibbsSteps_ = numSteps;
  return this;
}

El::Matrix<double> LDAModel::sgldEstimate(const El::Matrix<double>& thetaRaw) {
    El::Matrix<double> theta = thetaRaw;
    theta.Resize(K, W);

    El::Matrix<double> sgldEstimate(K, W, true);
    El::Zeros(sgldEstimate, K, W);

    // TODO: don't do this truncation hack, wrap around gracefully
    auto miniBatch = this->X(
            El::ALL,
            El::IR(
                this->minibatchIter % this->X.Width(),
                std::min(this->minibatchIter + this->batchSize, this->X.Width())));
    this->minibatchIter = std::min(this->minibatchIter + this->batchSize, this->X.Width());
    this->minibatchIter %= this->X.Width();

    double perplexitySumOverDocs = 0.0;
    for (int d=0; d<miniBatch.Width(); ++d) {
        auto doc_count_vec = miniBatch(El::ALL, d);

        int num_words_in_doc = 0;
        for (int i=0; i<doc_count_vec.Height(); ++i) {
            num_words_in_doc += doc_count_vec(i);
        }

        // Topic assignment of word i in current document d
        El::Matrix<int> index_to_topic(num_words_in_doc, 1, true);

        // Counts accumulator, topic_counts(k) = \sum_i 1{index_to_topic(i) == k}
        El::Matrix<int> topic_counts(K, 1, true);
        El::Zeros(topic_counts, K, 1);

        // Initialize topic assignments uniformly
        // TODO: use pi_k??
        for (int i=0; i<index_to_topic.Height(); ++i) {
            index_to_topic(i) = rand() % K;
            topic_counts(index_to_topic(i)) += 1;
        }

        // Perform sequential Gibbs sampling
        // Estimates sample of index_to_topic from posterior given theta
        gibbsSample(doc_count_vec, theta, index_to_topic, topic_counts);

        // Use sample to approximate stochastic gradient of log posterior
        El::Matrix<double> ones(theta.Width(), 1, true);
        El::Ones(ones, theta.Width(), 1);
        El::Matrix<double> theta_sum_over_w(theta.Height(), 1, true);
        El::Ones(theta_sum_over_w, theta.Height(), 1);
        El::Gemv(El::Orientation::NORMAL, 1.0, theta, ones, 0.0, theta_sum_over_w);
        int index = 0;
        for (int w=0; w<W; ++w) {
            for (int k=0; k<K; ++k) {
                // number of times topic k contains word w
                int n_kw = 0;
                for (int j=0; j<doc_count_vec(w); ++j) {
                    if (index_to_topic(index) == k) {
                        n_kw += 1;
                    }
                    index += 1;
                }
                const double pi_kw = 1.0 * theta(k,w) / theta_sum_over_w(k);
                sgldEstimate(k, w) += 1.0 * n_kw - pi_kw * topic_counts(k);
            }
        }

        // Calculate perplexity
        const double perplexity = estimatePerplexity(theta, theta_sum_over_w, num_words_in_doc, topic_counts);
        perplexitySumOverDocs += perplexity;
    }
    // Store average perplexity over minibatch
    this->perplexities_.push_back(perplexitySumOverDocs / miniBatch.Width());

    sgldEstimate *= 1.0 / this->batchSize;

    sgldEstimate.Resize(thetaRaw.Height(), thetaRaw.Width());

    return sgldEstimate;
}

void LDAModel::gibbsSample(
        const El::Matrix<int>& doc_count_vec,
        const El::Matrix<double>& theta,
        El::Matrix<int>& index_to_topic,
        El::Matrix<int>& topic_counts) const {
    for (int gibbsIter=0; gibbsIter < this->numGibbsSteps_; ++gibbsIter) {
        int index = 0;
        for (int w=0; w<W; ++w) {
          // iterate ovr words rather than document indices
          // allows re-use of posterior probability computations
          double posterior_probs[K]; // unnormalized, since discrete_distribution can take weights
          for (int k=0; k<K; ++k) {
            posterior_probs[k] = El::SafeAbs((this->alpha_ + topic_counts(k)) * theta(k, w));
          }
          for (int j=0; j<doc_count_vec(w); ++j) {
            const int old_topic_assignment = index_to_topic(index);

            // form cavity distribution after removing assignment i
            topic_counts(old_topic_assignment) -= 1;
            posterior_probs[old_topic_assignment] =
              El::SafeAbs((this->alpha_ + topic_counts(old_topic_assignment)) * theta(old_topic_assignment, w));

            // resample word's topic assignment
            gsl_ran_discrete_t * posterior = gsl_ran_discrete_preproc(K, &posterior_probs[0]);
            const int new_topic_assignment = gsl_ran_discrete(this->rng, posterior);
            index_to_topic(index) = new_topic_assignment;
            gsl_ran_discrete_free(posterior);

            // update posterior with new assignment
            topic_counts(new_topic_assignment) += 1;
            posterior_probs[new_topic_assignment] =
              El::SafeAbs((this->alpha_ + topic_counts(new_topic_assignment)) * theta(new_topic_assignment, w));

            index += 1;
          }
        }
    }
}

// Estimates the perplexity using a single posterior sample
// represented by `topic_counts`
double LDAModel::estimatePerplexity(
    const El::Matrix<double>& theta,
    const El::Matrix<double>& theta_sum_over_w,
    const int num_words_in_doc,
    const El::Matrix<int>& topic_counts) const {
  double sum_log_lik = 0;
  for (int w=0; w<W; ++w) {
    // probability of word over mixture of topics
    double word_prob = 0.0;
    for (int k=0; k<K; ++k) {
      const double eta = (1.0 * topic_counts(k) + this->alpha_) / (1.0 * num_words_in_doc + K * this->alpha_);
      const double pi_kw = 1.0 * theta(k,w) / theta_sum_over_w(k);
      word_prob += eta * pi_kw;
    }
    sum_log_lik += El::Log(word_prob);
  }
  // TODO: take exp, this is the log perplexity
  return -1.0 * sum_log_lik / num_words_in_doc;
}



El::Matrix<double> LDAModel::nablaLogPrior(const El::Matrix<double>& theta) const {
  El::Matrix<double> nablaLogPrior(theta.Height(), theta.Width(), true);

  El::Ones(nablaLogPrior, theta.Height(), theta.Width());
  nablaLogPrior *= this->beta_;
  nablaLogPrior -= theta;

  return nablaLogPrior;
}

void LDAModel::writePerplexities(const string& filename) {
  El::Matrix<double> mat(this->perplexities_.size(), 1, this->perplexities_.data(), 1);
  El::Write(mat, filename, El::MATRIX_MARKET);
  this->perplexities_.clear();
}

template class SGLDModel<double, int>;

} // namespace dsgld
