#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace libtea::ml {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct SplitData {
    Matrix X_train, X_test;
    Vector y_train, y_test;
};

inline SplitData train_test_split(
    const Matrix& X,
    const Vector& y,
    double test_size = 0.2,
    int random_seed = 42)
{
    size_t n = X.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t test_count = static_cast<size_t>(n * test_size);

    SplitData split;
    for (size_t i = 0; i < n; i++) {
        if (i < test_count) {
            split.X_test.push_back(X[indices[i]]);
            split.y_test.push_back(y[indices[i]]);
        } else {
            split.X_train.push_back(X[indices[i]]);
            split.y_train.push_back(y[indices[i]]);
        }
    }
    return split;
}

} // namespace libtea::ml