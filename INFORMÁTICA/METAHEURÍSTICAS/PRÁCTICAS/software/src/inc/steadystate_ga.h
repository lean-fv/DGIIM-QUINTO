#pragma once
#include "geneticalgorithm.h"

/**
 * Steady-state Genetic Algorithm (AGE).
 *
 * Each iteration produces two offspring and consumes at most two evaluations:
 *   1. Two parents are selected independently by tournament.
 *   2. The parents are always crossed; fix() is called on each offspring.
 *   3. Each offspring is independently mutated with probability Pm;
 *      fix() is called after each mutation.
 *   4. Each offspring is evaluated and replaces the current population's worst
 *      individual only if it is strictly better (no elitism needed: the worst
 *      is never better than an equal).
 *
 * Crossover is unconditional (probability 1.0). The loop is structured
 * (no break): the while condition guarantees the budget allows at least one
 * evaluation per iteration; the second is guarded by an inner if-check.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class SteadyStateGA : public GeneticAlgorithm<tDomain> {
public:
    /**
     * Construct a SteadyStateGA with the given hyper-parameters.
     *
     * @param M  Population size.
     * @param Pm Mutation probability per offspring.
     * @param ct Crossover operator to use (ARITHMETIC or BLX_ALPHA).
     * @param k  Tournament size for parent selection (default 3).
     */
    SteadyStateGA(int M, float Pm, CrossoverType ct, int k = 3)
        : GeneticAlgorithm<tDomain>(M, 1.0f, Pm, ct, k) {}

    virtual ~SteadyStateGA() {}

    /**
     * Run the AGE until the evaluation budget is exhausted.
     *
     * Initialises the population (consuming population_size evaluations), then
     * runs steady-state iterations until evals >= maxevals. Each iteration
     * attempts two replacements; the second is skipped if the budget is
     * already exhausted after the first. The best solution encountered across
     * all iterations is returned.
     *
     * @param problem  The problem to optimize.
     * @param maxevals Maximum number of fitness evaluations allowed.
     * @return         Best solution found, its fitness, and the total evaluation count.
     */
    ResultMH<tDomain> optimize(Problem<tDomain>& problem, int maxevals) override {
        int evals = this->initPopulation(problem);

        int best_idx = this->bestIndex();
        tSolution<tDomain> best_sol = this->pop[best_idx];
        tFitness            best_fit = this->fitnesses[best_idx];

        while (evals < maxevals) {
            // --- 1. Select 2 parents ---
            int i1 = this->tournamentSelect();
            int i2 = this->tournamentSelect();

            // --- 2. Crossover (always, Pc = 1.0) ---
            auto [c1, c2] = this->applyCrossover(this->pop[i1], this->pop[i2]);
            problem.fix(c1);
            problem.fix(c2);

            // --- 3. Mutation — each child independently with prob Pm ---
            if (Random::get<float>(0.0f, 1.0f) < this->mutation_prob) {
                this->applyMutation(c1);
                problem.fix(c1);
            }
            if (Random::get<float>(0.0f, 1.0f) < this->mutation_prob) {
                this->applyMutation(c2);
                problem.fix(c2);
            }

            // --- 4. Evaluate c1 and replace worst if better ---
            tFitness f1 = problem.fitness(c1);
            evals++;
            int w1 = this->worstIndex();
            if (f1 > this->fitnesses[w1]) {
                this->pop[w1]       = c1;
                this->fitnesses[w1] = f1;
                if (f1 > best_fit) { best_sol = c1; best_fit = f1; }
            }

            // --- 5. Evaluate c2 and replace worst if better ---
            if (evals < maxevals) {
                tFitness f2 = problem.fitness(c2);
                evals++;
                int w2 = this->worstIndex();
                if (f2 > this->fitnesses[w2]) {
                    this->pop[w2]       = c2;
                    this->fitnesses[w2] = f2;
                    if (f2 > best_fit) { best_sol = c2; best_fit = f2; }
                }
            }
        }

        return ResultMH<tDomain>(best_sol, best_fit, evals);
    }
};
