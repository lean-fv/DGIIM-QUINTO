#pragma once
#include "mh.h"

using namespace std;

/**
 * Implementation of the Random Search metaheuristic
 *  - Randomly generates solutions and selects the best one
 *
 * @see MH
 * @see Problem
 */
// Instanciamos la plantilla con el tipo que nos interese
using MHFloat = MH<float>;
using ProblemFloat = Problem<float>;
using ResultMHFloat = ResultMH<float>;

class GreedySearch : public MHFloat {

public:
    GreedySearch() : MH() {}
    virtual ~GreedySearch() {}
    // Implement the MH interface methods
    /**
     * Create random solutions until maxevals has been achieved, and returns the
     * best one.
     *
     * @param problem The problem to be optimized
     * @param maxevals Maximum number of evaluations allowed
     * @return A pair containing the best solution found and its fitness
     */
    virtual ResultMH<float> optimize(Problem<float> &problem, int maxevals) override;

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
    virtual float heuristic(const vector<float> &logarithmic_sum, const vector<vector<float>> &covariance_matrix, int asset_index,
                        float lambda);

};
