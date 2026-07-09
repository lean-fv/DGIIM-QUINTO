#pragma once
#include "generational_ga.h"
#include "localsearch.h"
#include <numeric>

/**
 * Abstract base class for Memetic Algorithms (AM).
 *
 * Extends GenerationalGA with a local-search (BL) phase applied periodically
 * every gen_period generations to a subset of the population defined by the
 * concrete subclass. BL evaluations count toward the global budget. If the
 * budget is exhausted before the scheduled BL step, that step is skipped.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class MemeticBase : public GenerationalGA<tDomain> {
protected:
    LocalSearch<tDomain> bl;   ///< Local search used for the BL phase.
    int bl_maxevals;           ///< Maximum evaluations given to BL per individual per call.
    int gen_period;            ///< Number of generations between successive BL applications.

    /**
     * Apply local search to a single individual in the population.
     *
     * Runs bl.optimize() starting from pop[i] with fitness fitnesses[i],
     * capped at min(bl_maxevals, remaining) to respect the global budget.
     * Updates pop[i] and fitnesses[i] in-place with the improved result.
     *
     * @param problem   The problem to optimize.
     * @param i         Index of the individual to improve.
     * @param remaining Evaluations left in the global budget.
     * @return          Number of fitness evaluations consumed by the local search.
     */
    int applyBLToOne(Problem<tDomain>& problem, int i, int remaining) {
        auto res = bl.optimize(problem, this->pop[i], this->fitnesses[i], remaining);
        this->pop[i]       = std::move(res.solution);
        this->fitnesses[i] = res.fitness;
        return res.evaluations;
    }

    /**
     * Apply local search to a subset of the population (defined by subclass).
     *
     * Called every gen_period generations if the budget is not yet exhausted.
     * Implementations must stop early when remaining reaches zero.
     *
     * @param problem   The problem to optimize.
     * @param remaining Evaluations left in the global budget at the time of the call.
     * @return          Total number of evaluations consumed across all BL calls.
     */
    virtual int applyBL(Problem<tDomain>& problem, int remaining) = 0;

public:
    /**
     * Construct a MemeticBase with the given hyper-parameters.
     *
     * @param M            Population size.
     * @param Pc           Crossover probability.
     * @param Pm           Mutation probability per individual.
     * @param ct           Crossover operator (ARITHMETIC or BLX_ALPHA).
     * @param k            Tournament size (default 3).
     * @param bl_ratio     Step-size ratio for the local search operator (default 0.15).
     * @param bl_maxevals  Maximum evaluations per individual per BL call (default 100).
     * @param gen_period   Generations between BL applications (default 10).
     */
    MemeticBase(int M=100, float Pc=0.8f, float Pm=0.1f, CrossoverType ct=CrossoverType::BLX_ALPHA, int k = 3,
                float bl_ratio = 0.15f, int bl_maxevals = 100, int gen_period = 10)
        : GenerationalGA<tDomain>(M, Pc, Pm, ct, k),
          bl(bl_ratio), bl_maxevals(bl_maxevals), gen_period(gen_period) {}

    virtual ~MemeticBase() {}

    /**
     * Run the memetic algorithm until the evaluation budget is exhausted.
     *
     * Initialises the population (consuming population_size evaluations), then
     * runs AGG generation cycles. Every gen_period generations, applyBL() is
     * called on the individual subset chosen by the subclass. The best solution
     * encountered throughout the entire run is returned.
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
        int generation = 0;

        while (evals < maxevals) {
            this->runOneGeneration(problem, maxevals, evals, best_sol, best_fit);
            generation++;

            // --- BL every gen_period generations ---
            if (generation % gen_period == 0 && evals < maxevals) {
                evals += applyBL(problem, min(bl_maxevals, maxevals - evals));
                int b = this->bestIndex();
                if (this->fitnesses[b] > best_fit) {
                    best_sol = this->pop[b];
                    best_fit = this->fitnesses[b];
                }
            }
        }

        return ResultMH<tDomain>(best_sol, best_fit,evals);
    }
};

/**
 * Memetic Algorithm — All variant (AM-All).
 *
 * At each scheduled BL phase, applies local search to every individual in
 * the population.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class AM_All : public MemeticBase<tDomain> {
protected:
    /**
     * Apply local search to all population_size individuals.
     *
     * Stops early if the remaining budget is exhausted between individuals.
     *
     * @param problem   The problem to optimize.
     * @param remaining Evaluations left in the global budget.
     * @return          Total evaluations consumed across all individuals.
     */
    int applyBL(Problem<tDomain>& problem, int remaining) override {
        int bl_evals = 0;
        for (int i = 0; i < this->population_size && remaining > 0; i++) {
            int used = this->applyBLToOne(problem, i, remaining);
            bl_evals  += used;
            remaining -= used;
        }
        return bl_evals;
    }
public:
    /**
     * Construct an AM_All with the given hyper-parameters.
     *
     * @param M            Population size.
     * @param Pc           Crossover probability.
     * @param Pm           Mutation probability per individual.
     * @param ct           Crossover operator (ARITHMETIC or BLX_ALPHA).
     * @param k            Tournament size (default 3).
     * @param bl_ratio     Step-size ratio for local search (default 0.15).
     * @param bl_maxevals  Maximum evaluations per BL call per individual (default 100).
     * @param gen_period   Generations between BL applications (default 10).
     */
    AM_All(int M=100, float Pc=0.8f, float Pm=0.1f, CrossoverType ct=CrossoverType::BLX_ALPHA, int k = 3,
           float bl_ratio = 0.15f, int bl_maxevals = 100, int gen_period = 10)
        : MemeticBase<tDomain>(M, Pc, Pm, ct, k, bl_ratio, bl_maxevals, gen_period) {}
};

/**
 * Memetic Algorithm — Random variant (AM-Rand).
 *
 * At each scheduled BL phase, applies local search to each individual
 * independently with probability bl_rand_prob.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class AM_Rand : public MemeticBase<tDomain> {
protected:
    float bl_rand_prob; ///< Probability of applying BL to each individual per scheduled phase.

    /**
     * Apply local search to each individual with probability bl_rand_prob.
     *
     * Stops early if the remaining budget is exhausted between individuals.
     *
     * @param problem   The problem to optimize.
     * @param remaining Evaluations left in the global budget.
     * @return          Total evaluations consumed.
     */
    int applyBL(Problem<tDomain>& problem, int remaining) override {
        int bl_evals = 0;
        for (int i = 0; i < this->population_size && remaining > 0; i++) {
            if (Random::get<float>(0.0f, 1.0f) < bl_rand_prob) {
                int used = this->applyBLToOne(problem, i, remaining);
                bl_evals  += used;
                remaining -= used;
            }
        }
        return bl_evals;
    }
public:
    /**
     * Construct an AM_Rand with the given hyper-parameters.
     *
     * @param M              Population size.
     * @param Pc             Crossover probability.
     * @param Pm             Mutation probability per individual.
     * @param ct             Crossover operator (ARITHMETIC or BLX_ALPHA).
     * @param k              Tournament size (default 3).
     * @param bl_ratio       Step-size ratio for local search (default 0.15).
     * @param bl_maxevals    Maximum evaluations per BL call per individual (default 100).
     * @param gen_period     Generations between BL applications (default 10).
     * @param bl_rand_prob   Probability of applying BL to each individual (default 0.1).
     */
    AM_Rand(int M=100, float Pc=0.8f, float Pm=0.1f, CrossoverType ct=CrossoverType::BLX_ALPHA, int k = 3,
            float bl_ratio = 0.15f, int bl_maxevals = 100, int gen_period = 10, float bl_rand_prob = 0.1f)
        : MemeticBase<tDomain>(M, Pc, Pm, ct, k, bl_ratio, bl_maxevals, gen_period),
          bl_rand_prob(bl_rand_prob) {}
};

/**
 * Memetic Algorithm — Best variant (AM-Best).
 *
 * At each scheduled BL phase, applies local search only to the top bl_top_ratio
 * fraction of individuals ranked by fitness. At least one individual is always
 * processed.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class AM_Best : public MemeticBase<tDomain> {
protected:
    float      bl_top_ratio; ///< Fraction of the population (by fitness rank) to apply BL to.
    vector<int> _idx;        ///< Scratch index buffer reused across applyBL() calls to avoid reallocation.

    /**
     * Apply local search to the top bl_top_ratio fraction of individuals by fitness.
     *
     * Sorts indices by descending fitness and processes the first
     * max(1, population_size * bl_top_ratio) of them. Stops early if the
     * remaining budget is exhausted between individuals.
     *
     * @param problem   The problem to optimize.
     * @param remaining Evaluations left in the global budget.
     * @return          Total evaluations consumed.
     */
    int applyBL(Problem<tDomain>& problem, int remaining) override {
        int bl_evals = 0;
        int top_n = max(1, static_cast<int>(this->population_size * bl_top_ratio));

        _idx.resize(this->population_size);
        iota(_idx.begin(), _idx.end(), 0);
        nth_element(_idx.begin(), _idx.begin() + top_n, _idx.end(), [&](int a, int b) {
            return this->fitnesses[a] > this->fitnesses[b];
        });

        for (int k = 0; k < top_n && remaining > 0; k++) {
            int used = this->applyBLToOne(problem, _idx[k], remaining);
            bl_evals  += used;
            remaining -= used;
        }
        return bl_evals;
    }
public:
    /**
     * Construct an AM_Best with the given hyper-parameters.
     *
     * @param M              Population size.
     * @param Pc             Crossover probability.
     * @param Pm             Mutation probability per individual.
     * @param ct             Crossover operator (ARITHMETIC or BLX_ALPHA).
     * @param k              Tournament size (default 3).
     * @param bl_ratio       Step-size ratio for local search (default 0.15).
     * @param bl_maxevals    Maximum evaluations per BL call per individual (default 100).
     * @param gen_period     Generations between BL applications (default 10).
     * @param bl_top_ratio   Fraction of top individuals to apply BL to (default 0.1).
     */
    AM_Best(int M=100, float Pc=0.8f, float Pm=0.1f, CrossoverType ct=CrossoverType::BLX_ALPHA, int k = 3,
            float bl_ratio = 0.15f, int bl_maxevals = 100, int gen_period = 10, float bl_top_ratio = 0.1f)
        : MemeticBase<tDomain>(M, Pc, Pm, ct, k, bl_ratio, bl_maxevals, gen_period),
          bl_top_ratio(bl_top_ratio) {}
};
