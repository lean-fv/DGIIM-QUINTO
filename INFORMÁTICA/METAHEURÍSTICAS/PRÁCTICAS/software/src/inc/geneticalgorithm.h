#pragma once
#include "mh.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

enum class CrossoverType { ARITHMETIC, BLX_ALPHA };

/**
 * Abstract base class for genetic algorithms.
 *
 * Provides shared operators (tournament selection, arithmetic crossover,
 * BLX-α crossover, and exchange mutation) and population management helpers.
 * Operators do NOT receive a Problem argument — callers are responsible for
 * calling problem.fix() after each crossover and mutation to repair
 * constraint violations.
 *
 * Concrete subclasses implement optimize() with their own generation scheme.
 *
 * @tparam tDomain Numeric type used for solution genes (e.g. float).
 */
template <typename tDomain>
class GeneticAlgorithm : public MH<tDomain> {
protected:
    int            population_size;  ///< Number of individuals in the population (M).
    float          crossover_prob;   ///< Probability of crossover per pair (Pc).
    float          mutation_prob;    ///< Probability of mutation per individual (Pm).
    float          blx_alpha;        ///< Exploration range parameter α for BLX-α crossover.
    float          mutation_ratio;   ///< Fraction of a gene's value transferred during exchange mutation.
    CrossoverType  crossover_type;   ///< Selects between ARITHMETIC and BLX_ALPHA crossover.
    int            tourney_k;        ///< Number of candidates drawn per tournament selection.

    const float EPSILON = 1e-8f;

    vector<tSolution<tDomain>> pop;            ///< Current population.
    vector<tFitness>           fitnesses;      ///< Fitness values corresponding to pop[].
    vector<tSolution<tDomain>> children_buf;   ///< Reusable generation buffer (avoids per-generation alloc).
    vector<tFitness>           child_fits_buf; ///< Reusable fitness buffer.

    /**
     * Select one individual by tournament.
     *
     * Draws tourney_k random candidates from the population and returns the
     * index of the one with the highest fitness. A larger k increases
     * selection pressure toward the best individuals.
     *
     * @return Index into pop[] of the tournament winner.
     */
    int tournamentSelect() const {
        int best = Random::get<int>(0, population_size - 1);
        for (int k = 1; k < tourney_k; k++) {
            int idx = Random::get<int>(0, population_size - 1);
            if (fitnesses[idx] > fitnesses[best])
                best = idx;
        }
        return best;
    }

    /**
     * Arithmetic crossover: each gene is a random convex combination of the parents.
     *
     * For each position i, a scalar r ~ U[0,1] is sampled independently and:
     *   o1[i] = r * p1[i] + (1-r) * p2[i]
     *   o2[i] = r * p2[i] + (1-r) * p1[i]
     *
     * Does NOT call fix() — caller is responsible for constraint repair.
     *
     * @param p1 First parent solution.
     * @param p2 Second parent solution.
     * @return   Pair of offspring solutions.
     */
    pair<tSolution<tDomain>, tSolution<tDomain>>
    crossoverArithmetic(const tSolution<tDomain>& p1, const tSolution<tDomain>& p2) const {
        size_t n = p1.size();
        tSolution<tDomain> o1(n), o2(n);
        for (size_t i = 0; i < n; i++) {
            float r = Random::get<float>(0.0f, 1.0f);
            o1[i] = static_cast<tDomain>(r * p1[i] + (1.0f - r) * p2[i]);
            o2[i] = static_cast<tDomain>(r * p2[i] + (1.0f - r) * p1[i]);
        }
        return {o1, o2};
    }

    /**
     * BLX-α crossover: each offspring gene is uniformly sampled from an extended interval.
     *
     * For each position i the interval [cmin - α*I, cmax + α*I] is built,
     * where cmin and cmax are the parental values and I = cmax - cmin.
     * Both offspring genes are independently drawn from this interval, allowing
     * exploration beyond the parental range when α > 0.
     *
     * Does NOT call fix() — caller is responsible for constraint repair.
     *
     * @param p1 First parent solution.
     * @param p2 Second parent solution.
     * @return   Pair of offspring solutions.
     */
    pair<tSolution<tDomain>, tSolution<tDomain>>
    crossoverBLX(const tSolution<tDomain>& p1, const tSolution<tDomain>& p2) const {
        size_t n = p1.size();
        tSolution<tDomain> o1(n), o2(n);
        for (size_t i = 0; i < n; i++) {
            float cmin = min(static_cast<float>(p1[i]), static_cast<float>(p2[i]));
            float cmax = max(static_cast<float>(p1[i]), static_cast<float>(p2[i]));
            float I    = cmax - cmin;
            float h_lo = cmin - blx_alpha * I;
            float h_hi = cmax + blx_alpha * I;
            if (h_lo > h_hi) h_lo = h_hi;  // guard: effolkronium requires lo <= hi
            o1[i] = static_cast<tDomain>(Random::get<float>(h_lo, h_hi));
            o2[i] = static_cast<tDomain>(Random::get<float>(h_lo, h_hi));
        }
        return {o1, o2};
    }

    /**
     * Dispatch crossover to the operator configured at construction time.
     *
     * Routes the call to crossoverArithmetic() or crossoverBLX() depending
     * on the crossover_type member.
     *
     * @param p1 First parent solution.
     * @param p2 Second parent solution.
     * @return   Pair of offspring solutions.
     */
    pair<tSolution<tDomain>, tSolution<tDomain>>
    applyCrossover(const tSolution<tDomain>& p1, const tSolution<tDomain>& p2) const {
        if (crossover_type == CrossoverType::ARITHMETIC) return crossoverArithmetic(p1, p2);

        return crossoverBLX(p1, p2);
    }

    /**
     * Exchange mutation: transfers a fraction of one gene's value to another.
     *
     * Picks two distinct random positions i and j. If sol[i] > EPSILON,
     * computes amount = mutation_ratio * sol[i], subtracts it from sol[i]
     * and adds it to sol[j], preserving the total sum of the solution.
     *
     * Does NOT call fix() — caller is responsible for constraint repair.
     *
     * @param sol Solution vector to mutate in-place.
     */
    void applyMutation(tSolution<tDomain>& sol) const {
        size_t n = sol.size();
        int i = Random::get<int>(0, static_cast<int>(n) - 1);
        int j;
        do { j = Random::get<int>(0, static_cast<int>(n) - 1); } while (j == i);
        if (sol[i] > EPSILON) {
            tDomain amount = static_cast<tDomain>(mutation_ratio) * sol[i];
            sol[i] -= amount;
            sol[j] += amount;
        }
    }

    /**
     * Initialise the population with random feasible solutions.
     *
     * Creates population_size individuals via problem.createSolution() and
     * evaluates each one immediately. Must be called once at the start of
     * optimize() before any generation cycle begins.
     *
     * @param problem The problem used to generate and evaluate individuals.
     * @return        Number of fitness evaluations consumed (= population_size).
     */
    int initPopulation(Problem<tDomain>& problem) {
        pop.resize(population_size);
        fitnesses.resize(population_size);
        for (int i = 0; i < population_size; i++) {
            pop[i]       = problem.createSolution();
            fitnesses[i] = problem.fitness(pop[i]);
        }
        return population_size;
    }

    /**
     * Return the index of the individual with the highest fitness.
     *
     * @return Index into pop[] of the best individual.
     */
    int bestIndex() const {
        return static_cast<int>(max_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin());
    }

    /**
     * Return the index of the individual with the lowest fitness.
     *
     * @return Index into pop[] of the worst individual.
     */
    int worstIndex() const {
        return static_cast<int>(min_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin());
    }

public:
    /**
     * Construct a genetic algorithm with the given hyper-parameters.
     *
     * @param M         Population size.
     * @param Pc        Crossover probability (fraction of pairs crossed per generation).
     * @param Pm        Mutation probability (fraction of individuals mutated per generation).
     * @param ct        Crossover operator to use (ARITHMETIC or BLX_ALPHA).
     * @param k         Tournament size for parent selection (default 3).
     * @param alpha     BLX-α exploration range parameter (default 0.3).
     * @param mut_ratio Fraction of a gene's value transferred during exchange mutation (default 0.15).
     */
    GeneticAlgorithm(int M, float Pc, float Pm, CrossoverType ct, int k = 3, float alpha = 0.3f, float mut_ratio = 0.15f)
        : population_size(M), crossover_prob(Pc), mutation_prob(Pm),
          blx_alpha(alpha), mutation_ratio(mut_ratio), crossover_type(ct),
          tourney_k(k) {}

    virtual ~GeneticAlgorithm() {}

    virtual ResultMH<tDomain> optimize(Problem<tDomain>& problem, int maxevals) override = 0;
};
