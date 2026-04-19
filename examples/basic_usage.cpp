#include <iostream>
#include "libtea/ml/linear_regression.hpp"
#include "libtea/ml/metrics.hpp"
#include "libtea/ml/train_test_split.hpp"

using namespace libtea::ml;

int main() {
    Matrix X = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}};
    Vector y  = {2,4,6,8,10,12,14,16,18,20};

    // Split 80/20
    auto split = train_test_split(X, y, 0.2, 42);

    // Train
    LinearRegression model(0.01, 1000);
    model.fit(split.X_train, split.y_train);

    // Evaluate on test set
    auto y_pred = model.predict(split.X_test);

    std::cout << "MAE:  " << mae(split.y_test, y_pred)      << "\n";
    std::cout << "MSE:  " << mse(split.y_test, y_pred)      << "\n";
    std::cout << "R²:   " << r2_score(split.y_test, y_pred) << "\n";

    return 0;
}