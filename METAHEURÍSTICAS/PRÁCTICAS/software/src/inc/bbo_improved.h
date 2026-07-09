#pragma once
#include "bbo.h"
#include <random>

/**
 * BBOImproved: Enhanced BBO with elitism, blended migration, and Gaussian mutation.
 *
 * Three targeted improvements over the base BBO, each addressing a known
 * weakness of the original algorithm:
 *
 *   1. Elitism
 *      The top `elitism_count` habitats are excluded from migration and
 *      mutation via getProcessCount(). They act as a persistent memory of
 *      the best solutions discovered so far, preventing regression.
 *
 *   2. Blended migration (BLX-style)
 *      Base BBO copies a SIV directly from the source habitat:
 *        habitat[j] = source[j]
 *      BBOImproved instead computes a convex combination:
 *        habitat[j] = alpha * source[j] + (1 - alpha) * habitat[j],  alpha ~ U(0,1)
 *      This preserves partial information from both habitats, producing more
 *      varied offspring while still propagating information from good sources.
 *
 *   3. Gaussian mutation
 *      Base BBO reinitialises a SIV uniformly anywhere in the domain:
 *        habitat[j] = U(domain.min, domain.max)
 *      BBOImproved instead adds a zero-mean Gaussian perturbation:
 *        habitat[j] += N(0, sigma),  sigma = gauss_sigma * (domain.max - domain.min)
 *      The result is clamped to the domain. This is a softer operator that
 *      explores the neighbourhood of the current value rather than jumping
 *      randomly across the entire domain.
 *
 * All three improvements are implemented by overriding getProcessCount(),
 * migratePopulation(), and mutatePopulation(). The optimize() loop, sorting,
 * and afterGeneration() hook are inherited unchanged from BBO.
 *
 * @tparam tDomain Numeric type for solution variables (float or double).
 */
template <typename tDomain>
class BBOImproved : public BBO<tDomain> {
private:
    int    elitism_count;
    double gauss_sigma;

protected:
    /**
     * @brief Return the number of non-elite habitats to process.
     *
     * Elite habitats (the top elitism_count after sorting) are shielded
     * from both migration and mutation, acting as persistent memory.
     *
     * @return int pop_size - elitism_count.
     */
    int getProcessCount() const override {
        return this->pop_size - elitism_count;
    }

    /**
     * @brief Perform one blended migration step on non-elite habitats.
     *
     * For each non-elite habitat i and each SIV j, replaces pop[i][j] with
     * a convex combination of the habitat's own value and the corresponding
     * SIV from a roulette-wheel selected emigrant in pop_copy, with
     * probability lambda[i]. Modified habitats are repaired and re-evaluated
     * immediately.
     *
     * @param problem   The optimisation problem (provides fix and fitness).
     * @param pop       Current population (modified in-place).
     * @param fit       Fitness values for pop (updated for modified habitats).
     * @param pop_copy  Pre-generation snapshot used as the emigrant source.
     * @param lambda    Immigration rates, one per habitat.
     * @param mu        Emigration rates used by the roulette-wheel selector.
     * @param sol_sz    Number of SIVs per habitat.
     * @param evals     Running evaluation counter (updated in-place).
     * @param maxevals  Maximum allowed evaluations (acts as a hard cap).
     */
    void migratePopulation(Problem<tDomain>& problem,
                           vector<tSolution<tDomain>>& pop, vector<tFitness>& fit,
                           const vector<tSolution<tDomain>>& pop_copy,
                           const vector<double>& lambda, const vector<double>& mu,
                           size_t sol_sz, int& evals, int maxevals) override {
        int count = getProcessCount();
        for (int i = 0; i < count && evals < maxevals; i++) {
            bool changed = false;
            for (size_t j = 0; j < sol_sz; j++) {
                if (Random::get<double>(0.0, 1.0) < lambda[i]) {
                    int    s     = this->rouletteWheel(mu);
                    double alpha = Random::get<double>(0.0, 1.0);
                    double old_j = static_cast<double>(pop[i][j]);
                    pop[i][j]    = static_cast<tDomain>(
                        alpha * static_cast<double>(pop_copy[s][j]) + (1.0 - alpha) * old_j);
                    changed = true;
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
     * @brief Perform one Gaussian mutation step on non-elite habitats.
     *
     * For each non-elite habitat i and each SIV j, adds a zero-mean
     * Gaussian perturbation with probability m_rate(i). The mutation rate
     * scales with how far habitat i's species count is from the most
     * probable count. The perturbed value is clamped to the domain.
     * Modified habitats are repaired and re-evaluated immediately.
     *
     * @param problem   The optimisation problem (provides fix and fitness).
     * @param pop       Current population (modified in-place).
     * @param fit       Fitness values for pop (updated for modified habitats).
     * @param sp        Species-count probabilities, one per habitat rank.
     * @param max_sp    Maximum value in sp (used to scale mutation rates).
     * @param domain    Allowed variable range {min, max} for clamping.
     * @param sol_sz    Number of SIVs per habitat.
     * @param evals     Running evaluation counter (updated in-place).
     * @param maxevals  Maximum allowed evaluations (acts as a hard cap).
     */
    void mutatePopulation(Problem<tDomain>& problem, vector<tSolution<tDomain>>& pop, 
                        vector<tFitness>& fit, const vector<double>& sp, double max_sp,
                          pair<tDomain, tDomain> domain, size_t sol_sz,
                          int& evals, int maxevals) override {
                            
        double range = static_cast<double>(domain.second) - static_cast<double>(domain.first);
        double sigma = gauss_sigma * range;
        int count = getProcessCount();
        for (int i = 0; i < count && evals < maxevals; i++) {
            double m_rate = this->max_mutation * (1.0 - sp[i] / max_sp);
            bool mutated = false;
            for (size_t j = 0; j < sol_sz; j++) {
                if (Random::get<double>(0.0, 1.0) < m_rate) {
                    double perturbed =
                        static_cast<double>(pop[i][j]) +
                        Random::get<std::normal_distribution<double>>(0.0, sigma);
                    perturbed = max(static_cast<double>(domain.first),
                                    min(static_cast<double>(domain.second), perturbed));
                    pop[i][j] = static_cast<tDomain>(perturbed);
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

public:
    /**
     * @brief Construct a BBOImproved instance with the given parameters.
     *
     * @param pop_size       Number of habitats in the population.
     * @param max_mutation   Maximum per-SIV mutation probability.
     * @param elitism_count  Number of top habitats shielded from modification.
     * @param gauss_sigma    Mutation standard deviation as a fraction of the domain range.
     */
    BBOImproved(int pop_size = 50, float max_mutation = 0.01f,
                int elitism_count = 2, double gauss_sigma = 0.1)
        : BBO<tDomain>(pop_size, max_mutation),
          elitism_count(elitism_count),
          gauss_sigma(gauss_sigma) {}

    virtual ~BBOImproved() {}
};
