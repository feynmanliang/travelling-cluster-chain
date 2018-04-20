#include <random>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "lda_model.h"

using std::vector;

namespace dsgld {

LDAModel::LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta)
    : SGLDModel<double, int>(X, K*X.Height())
    , alpha_(alpha)
    , beta_(beta)
    , numGibbsSteps_(100)
    , batchSize(50) // TODO: generalize to all of the models since they all use minibatching
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
    const int W = X.Height();
    const int K = thetaRaw.Height() / W;

    // TODO: move out to client
    const gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);

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
        auto doc = miniBatch(El::ALL, d);

        int num_words_in_doc = 0;
        for (int i=0; i<doc.Height(); ++i) {
            num_words_in_doc += doc(i);
        }

        // Topic assignment of word i in current document d. Denoted by z_{di} in literature
        El::Matrix<int> index_to_topic(num_words_in_doc, 1);

        // Count aggregate statistics
        El::Matrix<int> topic_counts;
        El::Zeros(topic_counts, K, 1);

        // Initialize topic assignments uniformly
        // TODO: use pi_k??
        for (int i=0; i<num_words_in_doc; ++i) {
            index_to_topic(i) = rand() % K;
            topic_counts(index_to_topic(i)) += 1;
        }

        // Perform sequential Gibbs sampling to get a draw from conditional z | w, \theta, \alpha
        for (int gibbsIter=0; gibbsIter < this->numGibbsSteps_; ++gibbsIter) {
            // do iteration over words rather than document indices to re-use posterior probability computations
            int offset = 0;
            for (int w=0; w<W; ++w) {
                double posterior_probs[K]; // unnormalized, since discrete_distribution can take weights
                for (int k=0; k<K; ++k) {
                    posterior_probs[k] = (this->alpha_ + topic_counts(k)) * theta(k, w);
                }
                for (int j=0; j<doc(w); ++j) {
                    const int i = offset + j; // absolute index of word in document
                    const int old_topic_assignment = index_to_topic(i);

                    // form cavity distribution after removing assignment i
                    topic_counts(old_topic_assignment) -= 1;
                    posterior_probs[old_topic_assignment] =
                        (this->alpha_ + topic_counts(old_topic_assignment)) * theta(old_topic_assignment, w);

                    // resample word's topic assignment
                    gsl_ran_discrete_t * posterior = gsl_ran_discrete_preproc(K, &posterior_probs[0]);
                    const int new_topic_assignment = gsl_ran_discrete(rng, posterior);
                    index_to_topic(i) = new_topic_assignment;
                    gsl_ran_discrete_free(posterior);


                    // update posterior with new assignment
                    topic_counts(new_topic_assignment) += 1;
                    posterior_probs[new_topic_assignment] =
                        (this->alpha_ + topic_counts(new_topic_assignment)) * theta(new_topic_assignment, w);
                }
                offset += doc(w);
            }
        }

        // Use sample to approximate stochastic gradient of log posterior
        El::Matrix<double> ones;
        El::Ones(ones, theta.Width(), 1);
        El::Matrix<double> denoms;
        El::Ones(denoms, theta.Width(), 1);
        El::Gemv(El::Orientation::NORMAL, 1.0, theta, ones, 0.0, denoms);
        int offset = 0;
        for (int w=0; w<W; ++w) {
            for (int k=0; k<K; ++k) {
                const double pi_kw = 1.0 * theta(k,w) / denoms(k);
                int n_dkw = 0;
                for (int j=0; j<doc(w); ++j) {
                    const int i = offset + j;
                    if (index_to_topic(i) == k) {
                        n_dkw += 1;
                    }
                }
                sgldEstimate(k, w) += 1.0 * n_dkw - pi_kw * topic_counts(k);
            }
            offset += doc(w);
        }

        // calculate perplexity
        double sum_log_lik = 0;
        for (int w=0; w<W; ++w) {
            double word_prob_acc = 0.0;
            for (int k=0; k<K; ++k) {
                const double eta = (1.0 * topic_counts(k) + this->alpha_) / (1.0 * num_words_in_doc + K * this->alpha_);
                const double pi_kw = 1.0 * theta(k,w) / denoms(k);
                word_prob_acc += eta * pi_kw;
            }
            sum_log_lik += El::Log(word_prob_acc);
        }
        // TODO: this is the log perplexity
        const double perplexity = -1.0 * sum_log_lik / num_words_in_doc;
        perplexitySumOverDocs += perplexity;
    }
    this->perplexities_.push_back(perplexitySumOverDocs / miniBatch.Width());

    sgldEstimate *= 1.0 / miniBatch.Width();

    sgldEstimate.Resize(thetaRaw.Height(), thetaRaw.Width());

    return sgldEstimate;
}

El::Matrix<double> LDAModel::nablaLogPrior(const El::Matrix<double>& theta) const {
    El::Matrix<double> nablaLogPrior;

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
