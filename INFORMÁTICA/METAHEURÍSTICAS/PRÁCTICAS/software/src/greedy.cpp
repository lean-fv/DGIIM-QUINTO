#include <cassert>
#include "greedy.h"
#include <iostream>
#include "portfolio.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;

template <class T> void print_vector(string name, const vector<T> &sol) {
    cout << name << ": ";

    for (auto elem : sol) {
        cout << elem << ", ";
    }
    cout << endl;
}

/**
 * Compute a heuristic score for a single asset.
 *
 * This function combines the asset's mean logarithmic return with a risk
 * penalty based on the asset's average covariance with all assets. The
 * `lambda` parameter controls the weight of the risk penalty: larger
 * values increase risk aversion by penalizing assets with higher average
 * covariance.
 *
 * @param logarithmic_sum Vector of mean logarithmic returns for each asset (size N).
 * @param covariance_matrix Covariance matrix (N x N) where covariance_matrix[i][j]
 *        represents the covariance between asset i and asset j.
 * @param asset_index Index of the asset (column) to evaluate.
 * @param lambda Risk-penalty coefficient (higher = more risk averse).
 * @return Heuristic value computed as: mean_log_return - lambda * mean_covariance.
 */

float GreedySearch::heuristic(const vector<float> &logarithmic_sum, const vector<vector<float>> &covariance_matrix,
                                int asset_index, float lambda) {
    
    size_t T = logarithmic_sum.size();
    size_t N = covariance_matrix.size();

    // 1. Compute the mean logarithmic return for the asset 'asset_index'
    float log_return = logarithmic_sum[asset_index];

    // 2. Compute the mean covariance of this asset with the rest
    float mean_cov = 0.0f;
    for (size_t j = 0; j < N; ++j) {
        mean_cov += covariance_matrix[j][asset_index];
    }
    mean_cov /= N;

    // 3. Return the heuristic value (return penalized by risk)
    return log_return - (lambda * mean_cov);
}

/**
 * Create random solutions until maxevals has been achieved, and returns the
 * best one.
 *
 * @param problem The problem to be optimized
 * @param maxevals Maximum number of evaluations allowed
 * @return A pair containing the best solution found and its fitness
 */
ResultMHFloat GreedySearch::optimize(Problem<float> &problem, int maxevals) {
    // 1. Cast the generic problem to our ProblemPortfolio
    ProblemPortfolio* port_problem = dynamic_cast<ProblemPortfolio*>(&problem);

    // 2. Extract required information via getters
    size_t size = port_problem->getSolutionSize();
    float max_val = port_problem->getMaxWeight();
    float lambda = port_problem->getLambda();
    const auto& log_sum = port_problem->getLogarithmicSum();
    const auto& cov_matrix = port_problem->getCovarianceMatrix();

    // Vector to store pairs <heuristic, asset_index>
    vector<pair<float, int>> heuristics_list;

    // 3. Compute heuristic for each asset using our function
    for (size_t i = 0; i < size; ++i) {
        float h_value = heuristic(log_sum, cov_matrix, i, lambda);
        heuristics_list.push_back({h_value, i});
    }

    // 4. Sort by descending heuristic
    sort(heuristics_list.begin(), heuristics_list.end(), 
            [](const pair<float, int>& a, const pair<float, int>& b) {
                return a.first > b.first; 
            });

    // 5. Build the solution following the Greedy rule
    tSolution<float> sol(size, 0.0f); // All zeros initially
    float sum_w = 0.0f;

    for (auto it = heuristics_list.begin(); it != heuristics_list.end() && sum_w < 1.0f - 1e-8f; ++it) {
        int idx = it->second; // Original asset index

        float remaining = 1.0f - sum_w;
        float assign = (remaining > max_val) ? max_val : remaining;
        sol[idx] = assign;
        sum_w += assign;
    }

    // Ensure solution respects bounds and sums to 1
    port_problem->fix(sol);

    // 6. Evaluate the constructed solution
    tFitness fitness = port_problem->fitness(sol);
    
    return ResultMHFloat(sol, fitness, 1);
}
