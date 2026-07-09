#pragma once
#include "geneticalgorithm.h"

/**
 * Generational Genetic Algorithm with elitism (AGG).
 *
 * Each generation proceeds as follows:
 *   1. The current best individual is saved (elite).
 *   2. M tournament draws build a mating pool of M parents.
 *   3. The first round(Pc * M / 2) consecutive parent pairs are crossed using
 *      the configured operator; remaining parents are copied unchanged.
 *      fix() is called on each offspring pair.
 *   4. round(Pm * M) random mutation events are applied to the child population;
 *      fix() is called after each mutation.
 *   5. All children are evaluated and replace the entire current population.
 *   6. If the best child is worse than the saved elite, the worst child is
 *      replaced by the elite (elitism guarantee).
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class GenerationalGA : public GeneticAlgorithm<tDomain> {
protected:
    /**
     * Execute one full generation cycle in-place.
     *
     * Performs selection, crossover, mutation, evaluation, elitism, and
     * population swap. Updates evals, best_sol, and best_fit if a new global
     * best is found. If the evaluation budget is exhausted mid-generation
     * during child evaluation, only the children evaluated so far are scored;
     * the rest receive -infinity and the elitism step still runs.
     *
     * @param problem  The problem being optimized.
     * @param maxevals Evaluation budget used as a hard cap during child evaluation.
     * @param evals    Running evaluation counter — updated in-place.
     * @param best_sol Global best solution — updated in-place if a new best is found.
     * @param best_fit Global best fitness — updated in-place if a new best is found.
     */
    void runOneGeneration(Problem<tDomain>& problem, int maxevals, int& evals, tSolution<tDomain>& best_sol, tFitness& best_fit) {
        int n_cross_pairs = static_cast<int>(
            round(this->crossover_prob * this->population_size / 2.0f));
        int n_mutations = static_cast<int>(
            round(this->mutation_prob * this->population_size));

        // --- Elite ---
        int elite_idx = this->bestIndex();
        tSolution<tDomain> elite_sol = this->pop[elite_idx];
        tFitness           elite_fit = this->fitnesses[elite_idx];

        // --- Selection: M tournament draws directly into reusable buffer ---
        this->children_buf.resize(this->population_size);
        for (int i = 0; i < this->population_size; i++)
            this->children_buf[i] = this->pop[this->tournamentSelect()];

        // --- Crossover: first n_cross_pairs consecutive pairs ---
        for (int p = 0; p < n_cross_pairs && 2 * p + 1 < this->population_size; p++) {
            auto [c1, c2] = this->applyCrossover(this->children_buf[2 * p], this->children_buf[2 * p + 1]);
            problem.fix(c1);
            problem.fix(c2);
            this->children_buf[2 * p]     = std::move(c1);
            this->children_buf[2 * p + 1] = std::move(c2);
        }

        // --- Mutation: n_mutations random events ---
        for (int m = 0; m < n_mutations; m++) {
            int idx = Random::get<int>(0, this->population_size - 1);
            this->applyMutation(this->children_buf[idx]);
            problem.fix(this->children_buf[idx]);
        }

        // --- Evaluate all children (stop if budget exhausted) ---
        this->child_fits_buf.assign(this->population_size, -numeric_limits<tFitness>::infinity());
        for (int i = 0; i < this->population_size && evals < maxevals; i++) {
            this->child_fits_buf[i] = problem.fitness(this->children_buf[i]);
            evals++;
        }

        // --- Elitism: if elite is better than the new best, replace the new best ---
        int best_child = static_cast<int>(max_element(this->child_fits_buf.begin(), this->child_fits_buf.end()) - this->child_fits_buf.begin());
        if (elite_fit > this->child_fits_buf[best_child]) {
            this->children_buf[best_child]   = elite_sol;
            this->child_fits_buf[best_child] = elite_fit;
        }

        // --- Swap population — preserves buffer capacity for next generation ---
        this->pop.swap(this->children_buf);
        this->fitnesses.swap(this->child_fits_buf);

        // --- Update global best ---
        int b = this->bestIndex();
        if (this->fitnesses[b] > best_fit) {
            best_sol = this->pop[b];
            best_fit = this->fitnesses[b];
        }
    }

public:
    /**
     * Construct a GenerationalGA with the given hyper-parameters.
     *
     * @param M  Population size.
     * @param Pc Crossover probability (fraction of pairs crossed per generation).
     * @param Pm Mutation probability (fraction of individuals mutated per generation).
     * @param ct Crossover operator to use (ARITHMETIC or BLX_ALPHA).
     * @param k  Tournament size for parent selection (default 3).
     */
    GenerationalGA(int M, float Pc, float Pm, CrossoverType ct, int k = 3)
        : GeneticAlgorithm<tDomain>(M, Pc, Pm, ct, k) {}

    virtual ~GenerationalGA() {}

    /**
     * Run the AGG until the evaluation budget is exhausted.
     *
     * Initialises the population (consuming population_size evaluations), then
     * runs generational cycles until evals >= maxevals. If the budget runs out
     * mid-generation during child evaluation, that generation is completed with
     * the children evaluated so far and the loop exits at the next while check.
     * The best solution encountered across all generations is returned.
     *
     * @param problem  The problem to optimize.
     * @param maxevals Maximum number of fitness evaluations allowed.
     * @return         Best solution found, its fitness, and the total evaluation count.
     */
    ResultMH<tDomain> optimize(Problem<tDomain>& problem, int maxevals) override {
        int evals = this->initPopulation(problem);

        int best_idx = this->bestIndex();
        tSolution<tDomain> best_sol = this->pop[best_idx];
        tFitness           best_fit = this->fitnesses[best_idx];

        while (evals < maxevals)
            runOneGeneration(problem, maxevals, evals, best_sol, best_fit);

        return ResultMH<tDomain>(best_sol, best_fit, evals);
    }
};
