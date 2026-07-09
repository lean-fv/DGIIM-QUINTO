#include "portfolio.h"
#include <string>
#include <vector>
#include <cmath>

/**
     * @brief Calculate arithmetic mean and logarithmic sum for each asset.
     *
        * Computes and stores the per-asset mean logarithmic returns in
        * the class member `logarithmic_sum` (i.e. the average of
        * `log(1 + return)` across periods for each asset). It also
        * returns the arithmetic mean `mu` used for covariance computation.
     *
     * @param historical_data Matrix of returns (rows = periods, cols = assets).
     * @return vector<float> Arithmetic mean (mu) for each asset.
     */
vector<float> ProblemPortfolio::calculateAverages(const vector<vector<float>> &historical_data) {
        
        size_t T = historical_data.size();
        vector<float> mu(size, 0.0f);
        
        // Initialize the class attribute for logarithmic sums
        logarithmic_sum.assign(size, 0.0f);

        // Accumulate the sums for both averages in a single pass for efficiency
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < size; ++i) {
                mu[i] += historical_data[t][i];
                logarithmic_sum[i] += log(1.0f + historical_data[t][i]);
            }
        }

        // Divide by T to get the final averages
        for (size_t i = 0; i < size; ++i) {
            mu[i] /= T;
        }

        return mu;
    }

    /**
     * @brief Compute the covariance matrix between asset returns.
     *
     * Fills the class member `covariance_matrix` with the estimated
     * covariance between every pair of assets using the provided arithmetic
     * means `mu`.
     *
     * @param historical_data Matrix of returns (rows = periods, cols = assets).
     * @param mu Arithmetic mean per asset.
     */
void ProblemPortfolio::calculateCovarianceMatrix(const vector<vector<float>> &historical_data, const vector<float> &mu) {
        
        size_t T = historical_data.size();
        // Initialize the class attribute matrix with zeros
        covariance_matrix.assign(size, vector<float>(size, 0.0f));

        for (size_t i = 0; i < size; ++i) {
            // Start j from i to only compute the upper triangular matrix (since it is symmetric)
            for (size_t j = i; j < size; ++j) {
                float cov_sum = 0.0f;
                
                // Sum the product of deviations from the mean
                for (size_t t = 0; t < T; ++t) {
                    cov_sum += (historical_data[t][i] - mu[i]) * (historical_data[t][j] - mu[j]);
                }

                float result = cov_sum / T;
                // Average the sum and mirror the result to the lower triangular matrix
                covariance_matrix[i][j] = result;
                covariance_matrix[j][i] = result;

            }
        }
    }

/**
 * @brief Construct a new ProblemPortfolio instance.
 *
 * Initializes the problem using historical return data and precomputes
 * the per-asset logarithmic sums and the covariance matrix used by the
 * fitness function. The first column (date) must be removed before
 * passing data to this constructor; `historical_data` should contain only
 * numeric returns with rows = periods and cols = assets.
 *
 * @param historical_data Matrix of returns (rows = periods, cols = assets).
 * @param min_w Minimum allowed weight per asset (box constraint lower bound).
 * @param max_w Maximum allowed weight per asset (box constraint upper bound).
 * @param lambda Risk-aversion parameter used in the fitness computation.
 */
ProblemPortfolio::ProblemPortfolio(const vector<vector<float>> &historical_data, const float &min_w, const float &max_w, float lambda) : Problem<float>() {
    min_weight = min_w;
    max_weight = max_w;
    this->lambda = lambda;

    if (historical_data.empty()) {
        size = 0;
    }
    else{

        // Extract dimensions
        size = historical_data[0].size();

        // 1. Compute both simple (mu) and logarithmic averages
        vector<float> mu = calculateAverages(historical_data);

        // 2. Compute the covariance matrix using the calculated simple average (mu)
        calculateCovarianceMatrix(historical_data, mu);
    }
}

/**
 * @brief Get the number of assets (solution size).
 *
 * @return size_t Number of variables in a solution vector.
 */
size_t ProblemPortfolio::getSolutionSize() { return size; }

/**
 * @brief Get the domain/bounds for each solution component.
 *
 * @return pair<float,float> Pair containing {min_weight, max_weight}.
 */
pair<float,float> ProblemPortfolio::getSolutionDomainRange() { return {min_weight, max_weight}; }

/**
 * @brief Get the precomputed per-asset logarithmic sums.
 *
 * These are used to compute expected logarithmic return for a portfolio.
 */
const vector<float>& ProblemPortfolio::getLogarithmicSum() const { return logarithmic_sum; }

/**
 * @brief Get the covariance matrix between asset returns.
 *
 * @return const vector<vector<float>>& Symmetric covariance matrix (size x size).
 */
const vector<vector<float>>& ProblemPortfolio::getCovarianceMatrix() const { return covariance_matrix; }

/**
 * @brief Get the risk-aversion parameter lambda.
 */
float ProblemPortfolio::getLambda() const { return lambda; }

/**
 * @brief Get the maximum allowed weight per asset.
 */
float ProblemPortfolio::getMaxWeight() const { return max_weight; }

/**
 * @brief Get the minimum allowed weight per asset.
 */
float ProblemPortfolio::getMinWeight() const { return min_weight; }


    /**
     * @brief Compute the average logarithmic return of a portfolio.
     *
     * Given a candidate solution (portfolio weights), this method computes the
     * weighted average of the per-asset logarithmic returns previously
     * computed by the problem instance. The result is normalized by the
     * number of assets so it represents an average return per asset.
     *
     * @param solution Portfolio weight vector (must have length equal to
     *                 `getSolutionSize()`).
     * @return float Average logarithmic return for the given portfolio.
     */
float ProblemPortfolio::getReturn(const tSolution<float> &solution) const {
    float return_value = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        return_value += solution[i] * logarithmic_sum[i];
    }

    return return_value;
}


    /**
     * @brief Compute portfolio risk.
     *
     * Calculates the portfolio risk using the precomputed covariance matrix
     * with the formula:
     *   risk = sum_i sum_j solution[i] * solution[j] * covariance_matrix[i][j]
     *
     * Notes:
     * - Uses the class member `covariance_matrix` (must be initialized).
     * - Time complexity is O(n^2), where n = `getSolutionSize()`.
     * - The provided `solution` vector must have length equal to `size`.
     *
     * @param solution Portfolio weight vector (length must be `getSolutionSize()`).
     * @return float Computed risk value (non-negative).
     */
float ProblemPortfolio::getRisk(const tSolution<float> &solution) const {
    float risk = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        risk += solution[i] * solution[i] * covariance_matrix[i][i];
        for (size_t j = i + 1; j < size; ++j) {
            risk += 2.0f * solution[i] * solution[j] * covariance_matrix[i][j];
        }
    }

    return sqrt(risk);
}


    /**
     * @brief Evaluate the fitness of a portfolio solution.
     *
     * The fitness is computed as the expected logarithmic return minus the
     * risk penalty: fitness = expected_return - (lambda * variance), where
     * `expected_return` is the weighted average of per-asset logarithmic
     * returns and `variance` is the portfolio variance computed using the
     * covariance matrix.
     *
     * @param solution Portfolio weight vector to evaluate (must sum to 1.0).
     * @return tFitness Higher values indicate better trade-offs between
     *         return and risk.
     */
tFitness ProblemPortfolio::fitness(const tSolution<float> &solution) {
    float expected_return = getReturn(solution);
    float risk = getRisk(solution);
    float fitness_value = expected_return - (lambda * risk);

    return static_cast<tFitness>(fitness_value);
}

    /**
     * @brief Create a random feasible portfolio solution.
     *
     * Generates a random vector of weights within [min_weight, max_weight],
     * randomly deactivates some assets, and then normalizes and repairs the
     * vector so that it sums to 1.0 and satisfies the box constraints.
     *
     * @return tSolution<float> A feasible solution vector.
     */
tSolution<float> ProblemPortfolio::createSolution() {
    tSolution<float> solution =  Random::get<vector>(min_weight, max_weight,size);

    int n_zeros = Random::get<int>(1, size-1);

    vector<int> indices(size);
    iota(indices.begin(), indices.end(), 0);

    Random::shuffle(indices);

    for (int i = 0; i < n_zeros; ++i) solution[indices[i]] = 0;

    fix(solution);

    return solution;
}

    /**
     * @brief Check whether a solution is valid for this problem.
     *
     * A solution is valid if all weights are within [min_weight, max_weight]
     * and the weights sum to 1.0 within a small tolerance.
     *
     * @param solution Candidate solution vector to validate.
     * @return true if the solution meets domain and budget constraints.
     * @return false otherwise.
     */
bool ProblemPortfolio::isValid(const tSolution<float> &solution) {
    float sum = accumulate(solution.begin(), solution.end(), 0.0f);
    bool in_range = true;
    for (size_t i = 0; i < size && in_range; ++i) {
        if (solution[i] < min_weight || solution[i] > max_weight) {
            in_range = false;
        }
    }
    return fabs(sum - 1.0f) < EPSILON && in_range;
}

    /**
     * @brief Repair a solution to satisfy budget and box constraints.
     *
     * This function enforces feasibility of a portfolio weight vector by:
     * 1) Handling the degenerate zero-sum case.
     * 2) Normalizing weights so they sum to 1.0.
     * 3) Applying box constraints: values below `min_weight` become inactive (0.0),
     *    values above `max_weight` are clamped to `max_weight`.
     * 4) Ensuring a minimum number of active assets (ceil(1 / max_weight)).
     * 5) Iteratively distributing any remaining difference (1.0 - sum) among
     *    candidate assets while respecting box constraints until the
     *    difference is within `EPSILON` or no distribution is possible.
     *
     * @param solution Solution vector to be modified in-place. After this call
     *                 the vector will (attempt to) satisfy box and budget constraints.
     */
void ProblemPortfolio::fix(tSolution<float> &solution) {

    float sum = accumulate(solution.begin(), solution.end(), 0.0f);

    // If all weights are zero avoid division by zero
    if (fabs(sum) < EPSILON) {
        solution[0] = 1.0f;
        sum = 1.0f;
    }

    // Normalize to make the weights sum to 1
    for (size_t i = 0; i < size; ++i) {
        solution[i] /= sum;
    }

    // Apply box constraints and count currently active assets
    int active_count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (solution[i] < min_weight) {
            solution[i] = 0.0f;
        } else {
            if (solution[i] > max_weight) solution[i] = max_weight;
            active_count++;
        }
    }

    // Minimum number of assets required to satisfy the upper-bound constraint
    int min_required_active = ceil(1.0f / max_weight);

    // If there are too few active assets, activate some at the minimum weight
    if (active_count < min_required_active) {
        for (size_t i = 0; i < size && active_count < min_required_active; ++i) {
            if (solution[i] < EPSILON) {
                solution[i] = min_weight;
                active_count++;
            }
        }
    }

    // Recompute the sum and the remaining difference to distribute
    sum = accumulate(solution.begin(), solution.end(), 0.0f);
    float dif = 1.0f - sum;
    bool possible_to_distribute = true;

    // Distribute difference
    vector<int> candidates;
    while (fabs(dif) > EPSILON && possible_to_distribute) {
        candidates.clear();

        for (size_t i = 0; i < size; ++i) {
            if ((dif > 0.0f && solution[i] > 0.0f && solution[i] < max_weight) ||
                (dif < 0.0f && solution[i] > min_weight)) {
                candidates.push_back(i);
            }
        }

        int values_to_distribute = candidates.size();

        if (values_to_distribute > 0) {
            float increment = dif / values_to_distribute;
            dif = 0.0f;

            for (size_t i : candidates) {
                solution[i] += increment;

                // If a candidate exceeds max, accumulate the overflow to dif
                // so it can be redistributed in the next iteration
                if (solution[i] > max_weight) {
                    dif += (solution[i] - max_weight);
                    solution[i] = max_weight;
                } else if (solution[i] < min_weight) {
                    // If it falls below min, accumulate the shortfall
                    dif += (solution[i] - min_weight);
                    solution[i] = min_weight;
                }
            }
        } else {
            possible_to_distribute = false;
        }
    }
}