#include <random>

#include "lda_model.h"

using std::vector;

namespace dsgld {

LDAModel::LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta)
    : SGLDModel<double, int>(X, K*X.Height()), alpha_(alpha), beta_(beta)
{
}

const int word_at_position(const int i, const El::Matrix<int>& doc_word_counts) {
    int w = 0;
    int sum = 0;
    while (sum + doc_word_counts(w) < i) {
        w += 1;
    }
    return w;
}

El::Matrix<double> LDAModel::sgldEstimate(const El::Matrix<double>& thetaRaw) const {

    // TODO: move out to client
    std::default_random_engine generator;

    El::Matrix<double> theta;
    theta.LockedAttach(theta.Height() / X.Height(), X.Height(), thetaRaw.LockedBuffer(), 0);
    const int K = theta.Width();
    const int W = theta.Height();

    El::Matrix<double> sgldEstimate(K, W, true);
    El::Zeros(sgldEstimate, K, W);

    auto miniBatch = this->X;

    // NOTE: uncomment to take single random sample as minibatch
    // leave commented to make cost of imbalance obvious
    /* miniBatch = miniBatch(El::ALL, El::IR(rand() % miniBatch.Width())); */
    for (int d=0; d<miniBatch.Width(); ++d) {
        auto doc = miniBatch(El::ALL, d);
        int num_words_in_doc = 0;
        for (int i=0; i<doc.Height(); ++i) {
            num_words_in_doc += doc(i);
        }

        El::Matrix<int> topic_counts;
        El::Zeros(topic_counts, K, 1);
        El::Matrix<int> word_index_topic_assignments(num_words_in_doc, 1);
        El::Zeros(word_index_topic_assignments, num_words_in_doc, 1);
        // TODO: make configurable
        for (int gibbsIter=0; gibbsIter < 1; ++gibbsIter) {
            El::Matrix<int> topic_counts0;
            El::Zeros(topic_counts0, topic_counts.Height(), topic_counts.Width());
            for (int i=0; i<num_words_in_doc; ++i) {
                // compute posterior probabilities
                // TODO: move outside of word iteration and do increments
                vector<double> posterior_probs; // unnormalized, since discrete_distribution can take weights
                for (int k=0; k<K; ++k) {
                    const int n_dk_less_i = word_index_topic_assignments(i) == k
                        ? topic_counts(k)
                        : topic_counts(k) - 1;
                    posterior_probs.push_back((this->alpha_ + n_dk_less_i) * theta(k, word_at_position(i, doc)));
                }

                // resample word's topic assignment
                std::discrete_distribution<> d(posterior_probs.begin(), posterior_probs.end());
                word_index_topic_assignments(i) = d(generator);

                topic_counts0(word_index_topic_assignments(i)) += 1;
            }
            topic_counts = topic_counts0;
        }
        // TODO: debugging, remove
        if (El::mpi::Rank() == 2)
            El::Print(topic_counts);

        // TODO: given topic posterior, contribute to estimator
        for (int k=0; k<K; ++k) {
            double denom = 0;
            for (int w=0; w<W; ++w) {
                denom += theta(k, w);
            }
            for (int w=0; w<W; ++w) {
                const double pi_kw = theta(k,w) / denom;
                int n_dkw = 0;
                for (int i=0; i<num_words_in_doc; ++i) {
                    if (word_at_position(i, doc) == w
                            &&word_index_topic_assignments(i) == k) {
                        n_dkw += 1;
                    }
                }
                sgldEstimate(k, w) += 1.0 * n_dkw - pi_kw * topic_counts(k);
            }
        }
    }

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

template class SGLDModel<double, int>;

} // namespace dsgld
