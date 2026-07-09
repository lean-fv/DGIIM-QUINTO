#pragma once
#include "bbo.h"
#include "soliswets.h"

/**
 * BBOSolisWets: Memetic BBO hybridised with the Solis-Wets local search.
 *
 * Extends BBO<tDomain> via the afterGeneration() hook: every sw_every
 * generations, a bounded Solis-Wets pass is applied to the current best
 * habitat (Lamarckian: the improved genotype replaces the original in-place).
 *
 * Unlike BBOLocalSearch (which uses portfolio weight-transfer moves),
 * SolisWets is domain-agnostic — it only calls problem.fitness() and
 * problem.fix() — making this class suitable for any continuous real-valued
 * problem, including CEC2017.
 *
 * Division of labour:
 *   BBO        — global exploration via migration and mutation.
 *   SolisWets  — local exploitation via adaptive random perturbations with
 *                directional bias memory and self-adjusting step size.
 *
 * @tparam tDomain Numeric type for solution variables (double for CEC2017).
 */
template <typename tDomain>
class BBOSolisWets : public BBO<tDomain> {
private:
    SolisWets<tDomain> sw;
    int sw_every;     // Apply SW once every this many BBO generations
    int sw_max_evals; // Max evaluations to spend per SW invocation
    int gen_count = 0;

protected:
    /**
     * @brief Apply Solis-Wets local search to the best habitat every sw_every generations.
     *
     * Increments the generation counter and, when it is a multiple of
     * sw_every, runs a bounded Solis-Wets pass on pop[N-1] (the current
     * best habitat). If the result is better than the current best, the
     * population entry is updated in-place (Lamarckian replacement).
     * Has no effect when the remaining evaluation budget is zero.
     *
     * @param problem  The optimisation problem.
     * @param pop      Current sorted population; pop[N-1] is the best habitat.
     * @param fit      Fitness values for pop; fit[N-1] is the best fitness.
     * @param evals    Running evaluation counter (incremented by SW calls).
     * @param maxevals Maximum allowed evaluations (caps the SW budget).
     */
    void afterGeneration(Problem<tDomain>& problem,
                          vector<tSolution<tDomain>>& pop,
                          vector<tFitness>& fit,
                          int& evals, int maxevals) override {
        gen_count++;
        if (gen_count % sw_every == 0) {
            int budget = min(sw_max_evals, maxevals - evals);
            if (budget > 0) {
                int N    = static_cast<int>(pop.size());
                auto res = sw.optimize(problem, pop[N - 1], fit[N - 1], budget);
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
     * @brief Construct a BBOSolisWets instance with the given parameters.
     *
     * @param pop_size      BBO population size.
     * @param max_mutation  BBO maximum per-SIV mutation probability.
     * @param sw_delta      Solis-Wets initial step size.
     * @param sw_every      Frequency of SW applications in BBO generations.
     * @param sw_max_evals  Evaluation budget per SW invocation.
     */
    BBOSolisWets(int pop_size = 50, float max_mutation = 0.01f,
                 double sw_delta = 0.2, int sw_every = 5, int sw_max_evals = 200)
        : BBO<tDomain>(pop_size, max_mutation),
          sw(sw_delta),
          sw_every(sw_every),
          sw_max_evals(sw_max_evals) {}

    virtual ~BBOSolisWets() {}

    /**
     * @brief Run BBOSolisWets until the evaluation budget is exhausted.
     *
     * Resets the generation counter, then delegates to the base BBO loop
     * which calls afterGeneration() after each generation to apply SW.
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
