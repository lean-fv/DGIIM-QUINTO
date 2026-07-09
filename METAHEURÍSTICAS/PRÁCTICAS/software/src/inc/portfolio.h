#pragma once
#include "problem.h"

using namespace std;

/**
 * ProblemPortfolio
 *
 * Represents a portfolio optimization problem constructed from historical
 * return data. The class computes per-asset logarithmic averages and the
 * covariance matrix used to evaluate portfolio risk. It provides fitness
 * evaluation and utility methods to create and repair candidate solutions
 * that satisfy box constraints and the budget (weights sum to 1).
 */
class ProblemPortfolio : public Problem<float> {
private:
    size_t size;

    vector<float> logarithmic_sum;
    vector<vector<float>> covariance_matrix;
    
    float lambda;
    const float EPSILON = 1e-8f;

    float min_weight;
    float max_weight;

    /**
     * @brief Calculate arithmetic mean and logarithmic sum for each asset.
     *
        * Computes and stores the per-asset mean logarithmic returns in
        * the class member `logarithmic_sum` (i.e. the average of
        * `log(1 + return)` across periods for each asset). It also
        * returns the arithmetic mean `mu` used for covariance computation.
     * @param historical_data Matrix of returns (rows = periods, cols = assets).
     * @return vector<float> Arithmetic mean (mu) for each asset.
     */
    vector<float> calculateAverages(const vector<vector<float>> &historical_data);

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
    void calculateCovarianceMatrix(const vector<vector<float>> &historical_data, const vector<float> &mu);

public:
    /**
     * @brief Construct a new ProblemPortfolio from parsed historical returns.
     *
     * The constructor expects `historical_data` where each row corresponds to
     * a time period and each column corresponds to an asset's return. The
     * first column (date) must be removed before calling this constructor.
     * It precomputes the per-asset logarithmic averages and the covariance
     * matrix used by the fitness function.
     *
     * @param historical_data Parsed returns matrix (rows = periods, cols = assets).
     * @param min_w Minimum allowed weight per asset.
     * @param max_w Maximum allowed weight per asset.
     * @param lambda Risk-aversion parameter.
     */
    ProblemPortfolio(const vector<vector<float>> &historical_data, const float &min_w, const float &max_w, float lambda);

    // Getters
    virtual size_t getSolutionSize() override;

    virtual pair<float, float> getSolutionDomainRange() override;

    /**
     * @brief Get the logarithmic sums used in fitness.
     *
     * @return const vector<float>& Reference to the stored logarithmic sums.
     */
    const vector<float>& getLogarithmicSum() const;

    /**
     * @brief Get the covariance matrix of asset returns.
     *
     * @return const vector<vector<float>>& Reference to the covariance matrix.
     */
    const vector<vector<float>>& getCovarianceMatrix() const;

    /**
     * @brief Return the lambda risk-aversion parameter.
     *
     * @return float Lambda multiplier used to penalize variance in fitness.
     */
    float getLambda() const;

    /**
     * @brief Return the maximum allowed weight per asset.
     *
     * @return float Maximum per-asset weight.
     */
    float getMaxWeight() const;

    /**
     * @brief Return the minimum allowed weight per asset.
     *
     * @return float Minimum per-asset weight.
     */
    float getMinWeight() const;


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
    float getReturn(const tSolution<float> &solution) const;


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
    float getRisk(const tSolution<float> &solution) const;


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
    tFitness fitness(const tSolution<float> &solution) override;

    /**
     * @brief Create a random feasible portfolio solution.
     *
     * Generates a random vector of weights within [min_weight, max_weight],
     * randomly deactivates some assets, and then normalizes and repairs the
     * vector so that it sums to 1.0 and satisfies the box constraints.
     *
     * @return tSolution<float> A feasible solution vector.
     */
    tSolution<float> createSolution() override;

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
    virtual bool isValid(const tSolution<float> &solution) override;

    /**
     * @brief Repair a solution to satisfy budget and box constraints.
     *
     * The method performs the following steps:
     * 1. Normalize the vector so weights sum to 1.0 (handle zero-sum case).
     * 2. Clamp values outside [min_weight, max_weight], marking inactive assets.
     * 3. If necessary, activate additional assets to ensure feasibility.
     * 4. Redistribute the remaining difference iteratively among candidates
     *    until the sum is 1.0 within a small tolerance or no distribution
     *    is possible.
     *
     * @param solution Solution vector to be modified in-place.
     */
    virtual void fix(tSolution<float> &solution) override;
};