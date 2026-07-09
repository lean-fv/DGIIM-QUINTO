#pragma once
#include "mh.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

/**
 * BBO: Biogeography-Based Optimization (base implementation)
 *
 * Adapted from Dan Simon, "Biogeography-Based Optimization",
 * IEEE Trans. Evolutionary Computation, Vol. 12, No. 6, 2008.
 *
 * Each candidate solution is a "habitat". Good solutions (high HSI) have
 * high emigration rates; poor solutions have high immigration rates.
 * At each generation, getProcessCount() habitats are candidates for:
 *   - Migration: import SIVs from other habitats selected proportionally
 *     to their emigration rate (roulette wheel).
 *   - Mutation: reinitialise SIVs with a probability that is highest at
 *     rank extremes and lowest at the middle, following the steady-state
 *     binomial species-count distribution C(N-1,k)/2^(N-1).
 *
 * Subclasses can customise behaviour by overriding:
 *   - getProcessCount()   — to exclude elite habitats from modification.
 *   - migratePopulation() — to change the migration strategy (e.g. blended).
 *   - mutatePopulation()  — to change the mutation strategy (e.g. Gaussian).
 *   - afterGeneration()   — to inject post-generation operators (e.g. local search).
 *
 * @tparam tDomain Numeric type for solution variables (float or double).
 */
template <typename tDomain>
class BBO : public MH<tDomain> {
protected:
    int   pop_size;
    float max_mutation;

    const double EPSILON = 1e-14;

    mutable vector<tSolution<tDomain>> sort_buf_pop;
    mutable vector<tFitness>           sort_buf_fit;

    /**
     * @brief Return the number of habitats subject to migration and mutation.
     *
     * Base BBO processes all N habitats. Subclasses may override to exclude
     * elite habitats from modification.
     *
     * @return int Number of habitats to process (default: pop_size).
     */
    virtual int getProcessCount() const {
        return pop_size;
    }

    /**
     * @brief Compute steady-state species-count probabilities.
     *
     * Returns p[k] proportional to C(N-1, k), normalised so the values sum
     * to 1. Middle-ranked habitats have the highest probability and therefore
     * the lowest mutation rate; rank extremes have the lowest probability
     * and therefore the highest mutation rate.
     *
     * @return vector<double> Normalised species-count probability for each rank.
     */
    vector<double> speciesProbabilities() const {
        int N = pop_size;
        vector<double> p(N);
        p[0] = 1.0;
        for (int k = 1; k < N; k++)
            p[k] = p[k - 1] * static_cast<double>(N - k) / k;
        double sum = 0.0;
        for (auto x : p) sum += x;
        for (auto& x : p) x /= sum;
        return p;
    }

    /**
     * @brief Select one index by roulette-wheel on the given weights.
     *
     * Samples a uniform value in [0, total] and returns the first index
     * whose cumulative weight meets or exceeds the sampled value. Falls
     * back to a uniform draw when the total weight is effectively zero.
     *
     * @param weights Non-negative emigration (or immigration) rates, one per habitat.
     * @return int Index of the selected habitat.
     */
    int rouletteWheel(const vector<double>& weights) const {
        int n = static_cast<int>(weights.size());
        double total = 0.0;
        for (auto w : weights) total += w;

        int selected = n - 1;
        if (total < EPSILON) {
            selected = Random::get<int>(0, n - 1);
        } else {
            double r      = Random::get<double>(0.0, total);
            double cumsum = 0.0;
            bool   found  = false;
            for (int i = 0; i < n && !found; i++) {
                cumsum += weights[i];
                if (cumsum >= r) {
                    selected = i;
                    found    = true;
                }
            }
        }
        return selected;
    }

    /**
     * @brief Sort the population in ascending fitness order in-place.
     *
     * After the call, pop[0] / fit[0] is the worst individual and
     * pop[N-1] / fit[N-1] is the best. Uses pre-allocated mutable buffers
     * to avoid heap allocation on every call.
     *
     * @param pop Population of solutions to be reordered.
     * @param fit Fitness values corresponding to each individual in pop.
     */
    void sortPopulation(vector<tSolution<tDomain>>& pop, vector<tFitness>& fit) const {

        int N = static_cast<int>(pop.size());
        vector<int> order(N);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(),[&](int a, int b) { return fit[a] < fit[b]; });

        sort_buf_pop.resize(N);
        sort_buf_fit.resize(N);

        for (int i = 0; i < N; i++) {
            sort_buf_pop[i] = std::move(pop[order[i]]);
            sort_buf_fit[i] = fit[order[i]];
        }
        
        pop.swap(sort_buf_pop);
        fit.swap(sort_buf_fit);
    }

    /**
     * @brief Compute linear immigration and emigration rates for N habitats.
     *
     * After sorting (worst to best), habitat i receives:
     *   lambda[i] = 1 - i/N  (immigration rate; worst has highest value),
     *   mu[i]     = i/N      (emigration rate; best has highest value).
     *
     * @param N Number of habitats (population size).
     * @return pair<vector<double>, vector<double>> {lambda, mu} rate vectors.
     */
    pair<vector<double>, vector<double>> computeMigrationRates(int N) const {
        vector<double> lambda(N), mu(N);
        for (int i = 0; i < N; i++) {
            lambda[i] = 1.0 - static_cast<double>(i) / N;
            mu[i]     =       static_cast<double>(i) / N;
        }
        return {lambda, mu};
    }

    /**
     * @brief Perform one migration step on the population.
     *
     * Processes the first getProcessCount() habitats. For each habitat i
     * and each SIV j, replaces pop[i][j] with the corresponding SIV from
     * a roulette-wheel selected emigrant in pop_copy, with probability
     * lambda[i]. Modified habitats are repaired and re-evaluated immediately.
     *
     * Override in subclasses to change the migration strategy (e.g. blended).
     *
     * @param problem   The optimisation problem (provides fix and fitness).
     * @param pop       Current population (modified in-place).
     * @param fit       Fitness values for pop (updated for modified habitats).
     * @param pop_copy  Pre-generation snapshot used as the emigrant source.
     * @param lambda    Immigration rates, one per habitat.
     * @param mu        Emigration rates used by the roulette-wheel selector.
     * @param sol_sz    Number of SIVs (decision variables) per habitat.
     * @param evals     Running evaluation counter (updated in-place).
     * @param maxevals  Maximum allowed evaluations (acts as a hard cap).
     */
    virtual void migratePopulation(Problem<tDomain>& problem,
                                   vector<tSolution<tDomain>>& pop, vector<tFitness>& fit,
                                   const vector<tSolution<tDomain>>& pop_copy,
                                   const vector<double>& lambda, const vector<double>& mu,
                                   size_t sol_sz, int& evals, int maxevals) {

        int count = getProcessCount();
        for (int i = 0; i < count && evals < maxevals; i++) {
            bool changed = false;
            for (size_t j = 0; j < sol_sz; j++) {
                if (Random::get<double>(0.0, 1.0) < lambda[i]) {
                    pop[i][j] = pop_copy[rouletteWheel(mu)][j];
                    changed   = true;
                }
            }
            if (changed) {
                problem.fix(pop[i]);
                fit[i] = problem.fitness(pop[i]);
                evals++;
            }
        }
    }

    /**
     * @brief Perform one mutation step on the population.
     *
     * Processes the first getProcessCount() habitats. For each habitat i
     * and each SIV j, replaces pop[i][j] with a value sampled uniformly
     * from the problem domain with probability m_rate(i). The mutation rate
     * scales with how far habitat i's species count is from the most probable
     * count. Modified habitats are repaired and re-evaluated immediately.
     *
     * Override in subclasses to change the mutation strategy (e.g. Gaussian).
     *
     * @param problem   The optimisation problem (provides fix and fitness).
     * @param pop       Current population (modified in-place).
     * @param fit       Fitness values for pop (updated for modified habitats).
     * @param sp        Species-count probabilities, one per habitat rank.
     * @param max_sp    Maximum value in sp (used to normalise mutation rates).
     * @param domain    Allowed variable range {min, max} for sampling/clamping.
     * @param sol_sz    Number of SIVs per habitat.
     * @param evals     Running evaluation counter (updated in-place).
     * @param maxevals  Maximum allowed evaluations (acts as a hard cap).
     */
    virtual void mutatePopulation(Problem<tDomain>& problem,vector<tSolution<tDomain>>& pop, 
                                  vector<tFitness>& fit, const vector<double>& sp, double max_sp,
                                  pair<tDomain, tDomain> domain, size_t sol_sz,
                                  int& evals, int maxevals) {

        int count = getProcessCount();
        for (int i = 0; i < count && evals < maxevals; i++) {
            double m_rate = max_mutation * (1.0 - sp[i] / max_sp);
            bool mutated = false;
            for (size_t j = 0; j < sol_sz; j++) {
                if (Random::get<double>(0.0, 1.0) < m_rate) {
                    pop[i][j] = static_cast<tDomain>(
                        Random::get<double>(static_cast<double>(domain.first),
                                           static_cast<double>(domain.second)));
                    mutated = true;
                }
            }
            if (mutated) {
                problem.fix(pop[i]);
                fit[i] = problem.fitness(pop[i]);
                evals++;
            }
        }
    }

    /**
     * @brief Hook called once per generation after migration, mutation, and sorting.
     *
     * The population is sorted in ascending order (pop[N-1] is the current
     * best) when this method is called. Override in subclasses to inject
     * additional operators such as local search.
     *
     * @param problem  The optimisation problem.
     * @param pop      Current sorted population.
     * @param fit      Fitness values for pop.
     * @param evals    Running evaluation counter (may be incremented by overrides).
     * @param maxevals Maximum allowed evaluations.
     */
    virtual void afterGeneration(Problem<tDomain>& problem, vector<tSolution<tDomain>>& pop,
                                  vector<tFitness>& fit, int& evals, int maxevals) {
        (void)problem; (void)pop; (void)fit; (void)evals; (void)maxevals;
    }

public:
    /**
     * @brief Construct a BBO instance with the given population parameters.
     *
     * @param pop_size      Number of habitats in the population.
     * @param max_mutation  Maximum per-SIV mutation probability.
     */
    BBO(int pop_size = 50, float max_mutation = 0.01f)
        : pop_size(pop_size), max_mutation(max_mutation) {}

    virtual ~BBO() {}

    /**
     * @brief Run BBO until the evaluation budget is exhausted.
     *
     * Initialises the population, then runs generation cycles composed of
     * migration, mutation, sorting, and the afterGeneration hook until evals
     * reaches maxevals. Migration and mutation are virtual and dispatch to
     * the appropriate strategy for the concrete subclass. The best habitat
     * (pop[N-1] after the final sort) is returned.
     *
     * @param problem  The optimisation problem to solve.
     * @param maxevals Maximum number of fitness evaluations allowed.
     * @return ResultMH<tDomain> Best solution found, its fitness, and eval count.
     */
    ResultMH<tDomain> optimize(Problem<tDomain>& problem, int maxevals) override {
        int N       = pop_size;
        auto domain = problem.getSolutionDomainRange();

        // --- Initialisation ---
        vector<tSolution<tDomain>> pop(N);
        vector<tFitness> fit(N);
        int evals = 0;
        for (int i = 0; i < N && evals < maxevals; i++) {
            pop[i] = problem.createSolution();
            fit[i] = problem.fitness(pop[i]);
            evals++;
        }
        sortPopulation(pop, fit);

        // --- Pre-compute rates (fixed while population size is constant) ---
        vector<double> sp     = speciesProbabilities();
        double         max_sp = *max_element(sp.begin(), sp.end());
        auto [lambda, mu]     = computeMigrationRates(N);
        const size_t sol_sz   = pop[0].size();
        vector<tSolution<tDomain>> pop_copy(N);

        // --- Generation loop ---
        while (evals < maxevals) {
            pop_copy = pop;
            migratePopulation(problem, pop, fit, pop_copy, lambda, mu, sol_sz, evals, maxevals);
            mutatePopulation(problem, pop, fit, sp, max_sp, domain, sol_sz, evals, maxevals);
            sortPopulation(pop, fit);
            afterGeneration(problem, pop, fit, evals, maxevals);
        }

        return ResultMH<tDomain>(pop[N - 1], fit[N - 1], static_cast<unsigned>(evals));
    }
};
