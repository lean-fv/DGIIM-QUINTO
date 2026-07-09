#pragma once
#include "mh.h"
#include "localsearch.h"
#include "simulatedannealing.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>
#include <cmath>

using namespace std;

/**
 * Abstract base for Iterated Local Search (ILS) variants.
 *
 * Implements the common ILS framework:
 *   1. Generate a random initial solution and optimise it with the inner optimiser.
 *   2. Repeat (num_iters - 1) times: mutate the best solution found so far,
 *      optimise the mutated solution, update best if improved.
 *
 * The inner optimiser is provided by subclasses via `innerOptimize()`.
 * Each call receives at most evals_per_iter evaluations (capped by the
 * remaining global budget).
 *
 * Mutation operator: randomly pick max(1, round(n * mut_ratio)) positions and
 * shuffle their values among themselves, then call problem.fix().
 *
 * Parameters:
 *   mut_ratio        — fraction of positions perturbed during mutation (default 0.20)
 *   num_iters        — total number of inner-optimiser calls, including the first
 *                      random-start call (default 5)
 *   evals_per_iter   — fixed eval cap per inner call (default 2000)
 */
template <typename tDomain>
class ILSBase : public MH<tDomain> {
protected:
    float mut_ratio;
    int   num_iters;
    int   evals_per_iter;
    vector<size_t>  _idx;   ///< Scratch index buffer reused across mutate() calls to avoid reallocation.
    vector<tDomain> _vals;  ///< Scratch value buffer reused across mutate() calls to avoid reallocation.

    /**
     * Run the inner optimiser starting from a given solution.
     * Must handle max_evals == 0 gracefully (return initial with 0 evals).
     */
    virtual ResultMH<tDomain> innerOptimize(Problem<tDomain> &problem,
                                            const tSolution<tDomain> &initial,
                                            tFitness initial_fitness,
                                            int max_evals) = 0;

    void mutate(Problem<tDomain> &problem, tSolution<tDomain> &sol) {
        size_t n = sol.size();
        int num_mut = max(1, static_cast<int>(round(n * mut_ratio)));

        _idx.resize(n);
        iota(_idx.begin(), _idx.end(), 0);
        Random::shuffle(_idx.begin(), _idx.end());
        _idx.resize(num_mut);

        _vals.resize(num_mut);
        for (int k = 0; k < num_mut; ++k) _vals[k] = sol[_idx[k]];

        Random::shuffle(_vals.begin(), _vals.end());

        for (int k = 0; k < num_mut; ++k) sol[_idx[k]] = _vals[k];

        problem.fix(sol);
    }

public:
    ILSBase(float mut_ratio = 0.20f, int num_iters = 5, int evals_per_iter = 2000)
        : MH<tDomain>(), mut_ratio(mut_ratio), num_iters(num_iters),
          evals_per_iter(evals_per_iter) {}
    virtual ~ILSBase() {}

    ResultMH<tDomain> optimize(Problem<tDomain> &problem, int maxevals) override {
        const int max_inner_evals = evals_per_iter - 1;
        int total_evals = 0;

        // --- Iteration 0: random initial solution ---
        int remaining_iters = num_iters;
        int per_iter_budget  = (maxevals - total_evals) / remaining_iters;

        tSolution<tDomain> s0 = problem.createSolution();
        tFitness f0 = problem.fitness(s0);
        total_evals++;

        auto res = innerOptimize(problem, s0, f0, min(max_inner_evals, per_iter_budget - 1));
        total_evals += res.evaluations;

        tSolution<tDomain>  best     = res.solution;
        tFitness            best_fit = res.fitness;

        // --- Iterations 1..(num_iters-1): mutate best, optimise, update ---
        for (int iter = 0; iter < this->num_iters - 1 && total_evals < maxevals; ++iter) {
            remaining_iters = num_iters - 1 - iter;
            per_iter_budget  = (maxevals - total_evals) / remaining_iters;

            tSolution<tDomain> mutated = best;
            mutate(problem, mutated);

            tFitness mutated_fit = problem.fitness(mutated);
            total_evals++;

            int remaining = min(max_inner_evals, per_iter_budget - 1);
            if (remaining > 0) {
                auto res2 = innerOptimize(problem, mutated, mutated_fit, remaining);
                total_evals += res2.evaluations;
                if (res2.fitness > best_fit) {
                    best     = res2.solution;
                    best_fit = res2.fitness;
                }
            } else {
                if (mutated_fit > best_fit) {
                    best     = mutated;
                    best_fit = mutated_fit;
                }
            }
        }

        return ResultMH<tDomain>(best, best_fit, total_evals);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ILS_BL: inner optimiser = LocalSearch
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ILS with LocalSearch as the inner optimiser (ILS-BL).
 */
template <typename tDomain>
class ILS_BL : public ILSBase<tDomain> {
private:
    LocalSearch<tDomain> bl;

protected:
    ResultMH<tDomain> innerOptimize(Problem<tDomain> &problem,
                                    const tSolution<tDomain> &initial,
                                    tFitness initial_fitness,
                                    int max_evals) override {
        return (max_evals > 0) ? bl.optimize(problem, initial, initial_fitness, max_evals)
                               : ResultMH<tDomain>(initial, initial_fitness, 0);
    }

public:
    ILS_BL(float bl_ratio, float mut_ratio = 0.20f, int num_iters = 5, int evals_per_iter = 2000)
        : ILSBase<tDomain>(mut_ratio, num_iters, evals_per_iter), bl(bl_ratio) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// ILS_ES: inner optimiser = SimulatedAnnealing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ILS hybridised with Simulated Annealing as the inner optimiser (ILS-ES).
 */
template <typename tDomain>
class ILS_ES : public ILSBase<tDomain> {
private:
    SimulatedAnnealing<tDomain> es;

protected:
    ResultMH<tDomain> innerOptimize(Problem<tDomain> &problem,
                                    const tSolution<tDomain> &initial,
                                    tFitness initial_fitness,
                                    int max_evals) override {
        return (max_evals > 0) ? es.optimize(problem, initial, initial_fitness, max_evals)
                               : ResultMH<tDomain>(initial, initial_fitness, 0);
    }

public:
    ILS_ES(float es_ratio, float mu = 0.2f, float phi = 0.3f, float Tf = 1e-3f,
           float mut_ratio = 0.20f, int num_iters = 5, int evals_per_iter = 2000)
        : ILSBase<tDomain>(mut_ratio, num_iters, evals_per_iter), es(es_ratio, mu, phi, Tf) {}
};
