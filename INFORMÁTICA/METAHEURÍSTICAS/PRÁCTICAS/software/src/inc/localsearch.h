#pragma once
#include "mhtrayectory.h"
#include <limits>
#include <cassert>

using namespace std;

/**
 * Implementation of the Local Search metaheuristic (First Improvement)
 * - Explores the neighborhood SEQUENTIALLY.
 * - Accepts the first neighbor that improves the current solution.
 * - Uses factorized/incremental evaluation for performance.
 * - Follows strict structured programming principles.
 *
 * @see MHTrayectory
 * @see Problem
 */
template <typename tDomain> class LocalSearch : public MHTrayectory<tDomain> {

private:
    float ratio; // Ratio of the weight to move from i to j (40%)
    const float EPSILON=1e-8f; // Small threshold to avoid numerical issues
    vector<pair<size_t, size_t>> P; // All (i,j) pairs with i!=j, built once per problem size

    /**
     * @brief Populate P with all ordered pairs (i,j) with i≠j, if not already built for this size.
     *
     * Fills member vector P with every ordered pair of distinct indices in [0, size).
     * If P already contains exactly size*(size-1) entries the method does nothing,
     * making repeated calls across optimize() invocations cheap.
     *
     * @param size Number of assets (equals problem.getSolutionSize()).
     */
    void buildPairsIfNeeded(size_t size) {
        if (P.size() != size * (size - 1)) {
            P.clear();
            P.reserve(size * (size - 1));
            for (size_t i = 0; i < size; ++i)
                for (size_t j = 0; j < size; ++j)
                    if (i != j) P.emplace_back(i, j);
        }
    }

public:
    LocalSearch(float r) : MHTrayectory<tDomain>(), ratio(r) {}
    virtual ~LocalSearch() {}

    /**
     * Run the Local Search (First Improvement) algorithm starting from a given solution.
     *
     * The method explores positions sequentially and for each position
     * iterates domain values sequentially. It performs incremental (factorized)
     * fitness evaluation and accepts the first improving neighbor.
     *
     * @param problem Problem instance to optimize (Pointer).
     * @param current The initial solution to start exploring from.
     * @param fitness The fitness of the initial solution.
     * @param maxevals Maximum number of fitness evaluations.
     * @return ResultMH containing the best solution found, its fitness and
     * the number of evaluations performed.
     */
    ResultMH<tDomain> optimize(Problem<tDomain> &problem, const tSolution<tDomain> &current,
                               tFitness fitness, int maxevals) override {
        assert(maxevals > 0);

        tSolution<tDomain> best = current;
        tFitness best_fitness = fitness;
        int evals = 0;

        size_t size = problem.getSolutionSize();
        auto domain = problem.getSolutionDomainRange();

        // 1. Build (or reuse) the set P of pairs (i,j) where i != j
        buildPairsIfNeeded(size);

        const tDomain tRatio      = static_cast<tDomain>(ratio);
        const tDomain min_active  = domain.first - static_cast<tDomain>(EPSILON);

        bool improvement = true;

        while (improvement && evals < maxevals) {
            improvement = false;
            size_t start = Random::get<size_t>(0, P.size() - 1);

            // 2. Sequential Exploration
            for (size_t k = 0; k < P.size() && evals < maxevals && !improvement; ++k) {
                auto& [i, j] = P[(start + k) % P.size()];

                // 3. We only operate if asset 'i' has some weight to move
                if (best[i] > 0) {
                    tDomain amount_to_move = tRatio * best[i];
                    
                    // --- EDGE CASE: 'j' starts from zero ---
                    // If 'j' currently has no weight, it MUST receive at least 
                    // the minimum threshold (domain.first) to become active.
                    if (best[j] < EPSILON && amount_to_move < domain.first) {
                        amount_to_move = domain.first;
                    }

                    // --- EXACT BOUNDARIES CALCULATION ---
                    // How much can 'j' receive as a maximum without exceeding its upper limit?
                    tDomain max_can_receive = domain.second - best[j];
                    
                    // How much can 'i' cede as a maximum without falling below its lower limit?
                    tDomain max_can_cede = best[i] - domain.first;

                    // --- ADJUST TO THE MOST RESTRICTIVE LIMIT ---
                    amount_to_move=min(max_can_receive, amount_to_move); // Cap by the receiver (j is full)
                    amount_to_move=min(max_can_cede, amount_to_move);    // Cap by the

                    // - If the amount has been clipped to almost zero, don't waste an evaluation
                    // - If 'j' started from zero and, after applying the caps from 'i', 
                    // we couldn't gather the legal minimum required for 'j', the move is illegal
                    bool valid = !((amount_to_move < EPSILON) || (best[j] < EPSILON && amount_to_move < min_active));

                    if (valid) {
                        // 4. Apply move in-place, evaluate, revert if no improvement
                        tDomain orig_i = best[i], orig_j = best[j];
                        best[i] -= amount_to_move;
                        best[j] += amount_to_move;

                        tFitness current_fitness = problem.fitness(best);
                        evals++;

                        if (current_fitness > best_fitness) {
                            best_fitness = current_fitness;
                            improvement  = true;
                        } else {
                            best[i] = orig_i;
                            best[j] = orig_j;
                        }
                    }
                }
            }
        }
        
        return ResultMH<tDomain>(best, best_fitness, evals);
    }
};