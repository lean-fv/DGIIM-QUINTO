#pragma once
#include "bbo.h"
#include "localsearch.h"

/**
 * BBOLocalSearch: Memetic BBO hybridised with portfolio Local Search.
 *
 * Extends BBO<tDomain> via the afterGeneration() hook: every ls_every
 * generations a bounded Local Search pass is applied to the current best
 * habitat (Lamarckian: the improved genotype replaces the original in-place).
 *
 * Division of labour:
 *   BBO  — global exploration: migration spreads good SIVs across the
 *           population; mutation injects diversity.
 *   LS   — local exploitation: first-improvement weight-transfer moves
 *           refine the elite habitat within its neighbourhood.
 *
 * The LocalSearch neighbourhood transfers a fixed ratio of weight between
 * portfolio assets, so this class is most effective for portfolio problems
 * (tDomain = float). It compiles for any tDomain but LS efficacy depends
 * on the domain structure.
 *
 * @tparam tDomain Variable type (float for portfolio).
 */
template <typename tDomain>
class BBOLocalSearch : public BBO<tDomain> {
private:
    LocalSearch<tDomain> ls;
    int ls_every;     // Apply LS once every this many BBO generations
    int ls_max_evals; // Max evaluations to spend per LS invocation
    int gen_count = 0;

protected:
    /**
     * @brief Apply Local Search to the best habitat every ls_every generations.
     *
     * Increments the generation counter and, when it is a multiple of
     * ls_every, runs a bounded Local Search pass on pop[N-1] (the current
     * best habitat). If the LS result is better than the current best, the
     * population entry is updated in-place (Lamarckian replacement).
     * Has no effect when the remaining evaluation budget is zero.
     *
     * @param problem  The optimisation problem.
     * @param pop      Current sorted population; pop[N-1] is the best habitat.
     * @param fit      Fitness values for pop; fit[N-1] is the best fitness.
     * @param evals    Running evaluation counter (incremented by LS calls).
     * @param maxevals Maximum allowed evaluations (caps the LS budget).
     */
    void afterGeneration(Problem<tDomain>& problem,
                          vector<tSolution<tDomain>>& pop,
                          vector<tFitness>& fit,
                          int& evals, int maxevals) override {
        gen_count++;
        if (gen_count % ls_every == 0) {
            int budget = min(ls_max_evals, maxevals - evals);
            if (budget > 0) {
                int N    = static_cast<int>(pop.size());
                auto res = ls.optimize(problem, pop[N - 1], fit[N - 1], budget);
                evals += static_cast<int>(res.evaluations);
                if (res.fitness > fit[N - 1]) {
                    pop[N - 1] = res.solution;
                    fit[N - 1] = res.fitness;
                }
            }
        }
    }

public:
    /**
     * @brief Construct a BBOLocalSearch instance with the given parameters.
     *
     * @param pop_size      BBO population size.
     * @param max_mutation  BBO maximum per-SIV mutation probability.
     * @param ls_ratio      LocalSearch weight-transfer ratio (portfolio domain).
     * @param ls_every      Frequency of LS applications in BBO generations.
     * @param ls_max_evals  Evaluation budget per LS invocation.
     */
    BBOLocalSearch(int pop_size = 50, float max_mutation = 0.01f,
                   float ls_ratio = 0.4f, int ls_every = 5, int ls_max_evals = 200)
        : BBO<tDomain>(pop_size, max_mutation),
          ls(ls_ratio),
          ls_every(ls_every),
          ls_max_evals(ls_max_evals) {}

    virtual ~BBOLocalSearch() {}

    /**
     * @brief Run BBOLocalSearch until the evaluation budget is exhausted.
     *
     * Resets the generation counter, then delegates to the base BBO loop
     * which calls afterGeneration() after each generation to apply LS.
     *
     * @param problem  The optimisation problem to solve.
     * @param maxevals Maximum number of fitness evaluations allowed.
     * @return ResultMH<tDomain> Best solution found, its fitness, and eval count.
     */
    ResultMH<tDomain> optimize(Problem<tDomain>& problem, int maxevals) override {
        gen_count = 0;
        return BBO<tDomain>::optimize(problem, maxevals);
    }
};
