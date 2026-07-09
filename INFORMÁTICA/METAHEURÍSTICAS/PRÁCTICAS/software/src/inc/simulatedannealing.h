#pragma once
#include "mhtrayectory.h"
#include <cmath>
#include <cassert>
#include <vector>

using namespace std;

/**
 * Simulated Annealing (Enfriamiento Simulado) metaheuristic.
 *
 * Uses a modified Cauchy cooling schedule. The inner loop explores the
 * neighbourhood using the same transfer operator as LocalSearch (move
 * ratio*s[i] from position i to j), shuffling all (i,j) pairs without
 * repetition within each temperature step. Acceptance follows the
 * Boltzmann criterion for maximisation: always accept improvements;
 * accept regressions with probability exp(-delta/T).
 *
 * Parameters:
 *   ratio  — fraction of s[i] to transfer per move (mirrors LocalSearch)
 *   mu     — worsening ratio for T0 formula (default 0.2)
 *   phi    — acceptance probability at mu-worse solution (default 0.3)
 *   Tf     — final temperature (default 1e-3)
 */
template <typename tDomain>
class SimulatedAnnealing : public MHTrayectory<tDomain> {
private:
    float ratio;
    float mu;
    float phi;
    float Tf_param;
    const float EPSILON = 1e-8f;

    vector<pair<size_t, size_t>> P;

    void buildPairsIfNeeded(size_t size) {
        if (P.size() != size * (size - 1)) {
            P.clear();
            P.reserve(size * (size - 1));
            for (size_t i = 0; i < size; ++i)
                for (size_t j = 0; j < size; ++j)
                    if (i != j) P.emplace_back(i, j);
        }
    }

public:
    SimulatedAnnealing(float ratio, float mu = 0.2f, float phi = 0.3f, float Tf = 1e-3f)
        : MHTrayectory<tDomain>(), ratio(ratio), mu(mu), phi(phi), Tf_param(Tf) {}

    virtual ~SimulatedAnnealing() {}

    /**
     * Run Simulated Annealing starting from a given solution.
     *
     * @param problem   The problem to optimise.
     * @param current   Initial solution.
     * @param fitness   Fitness of the initial solution.
     * @param maxevals  Maximum number of fitness evaluations.
     * @return ResultMH with best solution found, its fitness, and eval count.
     */
    ResultMH<tDomain> optimize(Problem<tDomain> &problem, const tSolution<tDomain> &current,
                                tFitness fitness, int maxevals) override {
        assert(maxevals > 0);

        size_t m = problem.getSolutionSize();
        auto domain = problem.getSolutionDomainRange();
        buildPairsIfNeeded(m);

        int max_vecinos = 10 * static_cast<int>(m);
        int max_exitos  =      static_cast<int>(m);

        // --- Temperature initialisation (T0 formula from the guide) ---
        float T0 = mu * std::abs(fitness) / (-std::log(phi));
        float Tf = Tf_param;
        // Guard: degenerate case (near-zero fitness or T0 too small)
        if (T0 <= Tf || T0 <= EPSILON) T0 = Tf * 2.0f;

        int   M_steps = std::max(1, maxevals / max_vecinos);
        float beta    = (T0 - Tf) / (static_cast<float>(M_steps) * T0 * Tf);

        // --- State ---
        tSolution<tDomain> s        = current;
        tFitness           s_fit    = fitness;
        tSolution<tDomain> best     = current;
        tFitness           best_fit = fitness;
        int evals = 0;
        float T = T0;
        bool accepted = true;

        while (evals < maxevals && accepted) {
            size_t start = Random::get<size_t>(0, P.size() - 1);
            int vecinos = 0;
            int exitos  = 0;

            for (size_t k = 0; k < P.size() && vecinos < max_vecinos
                && exitos < max_exitos && evals < maxevals; ++k)
            {
                size_t i = P[(start + k) % P.size()].first;
                size_t j = P[(start + k) % P.size()].second;

                if (s[i] > EPSILON) {
                    tDomain amount = static_cast<tDomain>(ratio) * s[i];

                    // Boundary logic
                    if (s[j] < EPSILON && amount < domain.first)
                        amount = domain.first;

                    tDomain max_recv = domain.second - s[j];
                    tDomain max_cede = s[i] - domain.first;
                    amount = std::min(max_recv, amount);
                    amount = std::min(max_cede, amount);

                    bool valid = !((amount < EPSILON) ||
                                    (s[j] < EPSILON && amount < domain.first - EPSILON));
                    if (valid) {
                        s[i] -= amount;
                        s[j] += amount;

                        tFitness n_fit = problem.fitness(s);
                        evals++;
                        vecinos++;

                        // delta > 0 means regression (maximisation)
                        float delta = s_fit - n_fit;
                        if (delta < 0.0f || Random::get<float>(0.0f, 1.0f) < std::exp(-delta / T)) {
                            s_fit = n_fit;
                            exitos++;
                            if (s_fit > best_fit) {
                                best     = s;
                                best_fit = s_fit;
                            }
                        } else {
                            s[i] += amount;
                            s[j] -= amount;
                        }
                    }
                }
            }

            accepted = (exitos > 0);

            // Modified Cauchy cooling
            T = T / (1.0f + beta * T);
        }

        return ResultMH<tDomain>(best, best_fit, evals);
    }
};
