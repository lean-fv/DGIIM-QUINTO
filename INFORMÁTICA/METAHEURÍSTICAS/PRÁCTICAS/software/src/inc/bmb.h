#pragma once
#include "mh.h"
#include "localsearch.h"
#include <limits>

using namespace std;

/**
 * Basic Multi-start Search (Búsqueda Multiarranque Básica, BMB).
 *
 * Generates `restarts` independent random solutions and optimises each
 * with LocalSearch (up to `evals_per_restart` evaluations). Returns the
 * best solution found across all restarts.
 */
template <typename tDomain>
class BMB : public MH<tDomain> {
private:
    LocalSearch<tDomain> bl;
    int restarts;
    int evals_per_restart;

public:
    /**
     * @param bl_ratio          Transfer ratio for the inner LocalSearch.
     * @param restarts          Number of independent restarts (default 5).
     * @param evals_per_restart Max evaluations per restart, including the
     *                          initial fitness call (default 2000).
     */
    BMB(float bl_ratio, int restarts = 5, int evals_per_restart = 2000)
        : MH<tDomain>(), bl(bl_ratio),
          restarts(restarts), evals_per_restart(evals_per_restart) {}

    virtual ~BMB() {}

    ResultMH<tDomain> optimize(Problem<tDomain> &problem, int maxevals) override {
        tSolution<tDomain> best;
        tFitness best_fit = -numeric_limits<tFitness>::infinity();
        int total_evals = 0;
        const int max_ls_evals = evals_per_restart - 1;

        for (int r = 0; r < restarts && total_evals < maxevals; ++r) {
            
            int remaining_restarts = restarts - r;
            int per_restart_budget = (maxevals - total_evals) / remaining_restarts;
            
            tSolution<tDomain> s0 = problem.createSolution();
            tFitness f0 = problem.fitness(s0);
            total_evals++;

            int remaining = min(max_ls_evals, per_restart_budget - 1);
            if (remaining > 0) {
                auto res = bl.optimize(problem, s0, f0, remaining);
                total_evals += res.evaluations;
                if (res.fitness > best_fit) {
                    best     = std::move(res.solution);
                    best_fit = res.fitness;
                }
            } else {
                if (f0 > best_fit) {
                    best     = std::move(s0);
                    best_fit = f0;
                }
            }
        }

        return ResultMH<tDomain>(best, best_fit, total_evals);
    }
};
