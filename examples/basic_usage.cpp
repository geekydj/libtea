#include <iostream>
#include "libtea/ml/linear_regression.hpp"
#include "libtea/ml/metrics.hpp"

using namespace libtea::ml;

int main() {

    Matrix X = {
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5}
    };

    Vector y = {3, 5, 7, 9};

    LinearRegression model;

    model.fit(X, y);

    Vector pred = model.predict(X);

    std::cout << "MAE: " << mae(y, pred) << "\n";
    std::cout << "MSE: " << mse(y, pred) << "\n";

    std::cout << "\nPredictions:\n";
    for (double v : pred)
        std::cout << v << "\n";

    return 0;
}