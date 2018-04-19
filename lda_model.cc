#include <random>

#include "lda_model.h"

using std::vector;

namespace dsgld {

// Given a count vector, returns a mapping from position in document to word at that position
// Since LDA uses bag of words representation, we can just unroll the count vector sequentially
const El::Matrix<int> make_index_to_word(const El::Matrix<int>& doc) {
    int num_words_in_doc = 0;
    for (int i=0; i<doc.Height(); ++i) {
        num_words_in_doc += doc(i);
    }

    El::Matrix<int> index_to_word;
    El::Ones(index_to_word, num_words_in_doc, 1);

    int startIdx = 0;
    for (int w=0; w<doc.Height(); ++w) {
        index_to_word(El::IR(startIdx, startIdx + doc(w)), 0) *= w;
        startIdx += doc(w);
    }

    return index_to_word;
}

LDAModel::LDAModel(const El::Matrix<int>& X, const int K, const double alpha, const double beta)
    : SGLDModel<double, int>(X, K*X.Height()), alpha_(alpha), beta_(beta)
{
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
    miniBatch = miniBatch(El::ALL, El::IR(rand() % miniBatch.Width()));

    for (int d=0; d<miniBatch.Width(); ++d) {
        auto doc = miniBatch(El::ALL, d);

        const El::Matrix<int> index_to_word = make_index_to_word(doc);
        const int num_words_in_doc = index_to_word.Height();

        // Topic assignment of word i in current document d. Denoted by z_{di} in literature
        El::Matrix<int> word_to_topic(num_words_in_doc, 1);

        // Count aggregate statistics
        El::Matrix<int> topic_counts;
        El::Zeros(topic_counts, K, 1);

        // Initialize topic assignments uniformly
        // TODO: use pi_k??
        for (int i=0; i<num_words_in_doc; ++i) {
            word_to_topic(i) = rand() % K;
            topic_counts(word_to_topic(i)) += 1;
        }

        // Perform sequential Gibbs sampling to get a draw from conditional z | w, \theta, \alpha
        // TODO: make num iterationsconfigurable
        for (int gibbsIter=0; gibbsIter < 1; ++gibbsIter) {
            // do iteration over words rather than document indices (index_to_word) to re-use posterior probability computations
            int offset = 0;
            for (int w=0; w<W; ++w) {
                vector<double> posterior_probs; // unnormalized, since discrete_distribution can take weights
                for (int k=0; k<K; ++k) {
                    posterior_probs.push_back((this->alpha_ + topic_counts(k)) * theta(k, w));
                }
                for (int j=0; j<doc(w); ++j) {
                    const int i = offset + j; // absolute index of word in document
                    const int old_topic_assignment = word_to_topic(i);

                    // form cavity distribution after removing assignment i
                    posterior_probs[old_topic_assignment] =
                        (this->alpha_ + topic_counts(old_topic_assignment) - 1) * theta(old_topic_assignment, w);

                    // resample word's topic assignment
                    topic_counts(word_to_topic(i)) -= 1;
                    std::discrete_distribution<> d(posterior_probs.begin(), posterior_probs.end());
                    const int new_topic_assignment = d(generator);
                    word_to_topic(i) = new_topic_assignment;
                    topic_counts(new_topic_assignment) += 1;

                    // update posterior with new assignment
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
        El::Gemv(El::Orientation::NORMAL, 1.0, theta, ones, 0.0, denoms);
        for (int k=0; k<K; ++k) {
            for (int w=0; w<W; ++w) {
                const double pi_kw = theta(k,w) / denoms(k);
                int n_dkw = 0;
                for (int i=0; i<num_words_in_doc; ++i) {
                    if (index_to_word(i) == w && word_to_topic(i) == k) {
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
