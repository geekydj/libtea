#pragma once
#include <vector>
#include <cmath>
#include <numeric>

namespace libtea::ml {

using Vector = std::vector<double>;

inline double mae(const Vector& y_true, const Vector& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++)
        sum += std::abs(y_true[i] - y_pred[i]);
    return sum / y_true.size();
}

inline double mse(const Vector& y_true, const Vector& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double d = y_true[i] - y_pred[i];
        sum += d * d;
    }
    return sum / y_true.size();
}

inline double r2_score(const Vector& y_true, const Vector& y_pred) {
    double mean_y = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }

    return 1.0 - (ss_res / ss_tot);
}

} // namespace libtea::ml