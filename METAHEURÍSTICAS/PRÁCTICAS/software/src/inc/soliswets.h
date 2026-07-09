#pragma once
#include "mhtrayectory.h"
#include <vector>

using namespace std;

/**
 * SolisWets: Gradient-free continuous local search.
 *
 * Adapted from the reference implementation in testsolis.cc (CEC2017 library).
 * Original algorithm: Solis & Wets, "Minimization by random search techniques",
 * Mathematics of Operations Research, 6(1), 1981.
 *
 * Each iteration generates a random perturbation dif[i] ~ U(0, delta) and
 * tries two candidate moves from the current solution:
 *   1. sol + dif + bias   (positive step along accumulated bias direction)
 *   2. sol - dif - bias   (negative step, tried only if (1) fails)
 *
 * The bias vector remembers successful directions:
 *   - Positive step succeeds: bias reinforced toward dif  (increm_bias)
 *   - Negative step succeeds: bias reinforced away from dif (decrement_bias)
 *   - Both fail:              bias halved
 *
 * The step size delta adapts automatically:
 *   - Doubled after 5 consecutive successes (exploit promising direction)
 *   - Halved  after 3 consecutive failures  (refine around current point)
 *
 * Compared to the portfolio LocalSearch, this operator is domain-agnostic:
 * it only requires problem.fitness() and problem.fix(), making it suitable
 * for any continuous real-valued problem (e.g. CEC2017).
 *
 * @tparam tDomain Numeric type for solution variables (double for CEC2017).
 */
template <typename tDomain>
class SolisWets : public MHTrayectory<tDomain> {
private:
    double delta_init; // Initial step size (adapted automatically during search)

public:
    /**
     * @param delta Initial step size. The adaptive mechanism doubles it after
     *              5 successes and halves it after 3 failures. A good starting
     *              value is roughly 0.1–1% of the domain range.
     */
    explicit SolisWets(double delta = 0.2) : delta_init(delta) {}
    virtual ~SolisWets() {}

    ResultMH<tDomain> optimize(Problem<tDomain>& problem,
                                const tSolution<tDomain>& current,
                                tFitness start_fitness, int maxevals) override {
        int dim = static_cast<int>(current.size());

        tSolution<tDomain> sol(current);
        tFitness           fitness = start_fitness;

        vector<double> bias(dim, 0.0);
        vector<double> dif(dim);
        tSolution<tDomain> newsol(dim);

        double delta      = delta_init;
        int    evals      = 0;
        int    num_success = 0;
        int    num_failed  = 0;

        while (evals < maxevals) {
            // Build perturbation: dif[i] ~ U(0, delta)
            for (int i = 0; i < dim; i++) {
                dif[i]    = Random::get<double>(0.0, delta);
                newsol[i] = static_cast<tDomain>(
                    static_cast<double>(sol[i]) + dif[i] + bias[i]);
            }
            problem.fix(newsol);
            tFitness newfit = problem.fitness(newsol);
            evals++;

            if (newfit > fitness) {
                sol     = newsol;
                fitness = newfit;
                // increm_bias: reinforce direction
                for (int i = 0; i < dim; i++)
                    bias[i] = 0.2 * bias[i] + 0.4 * (dif[i] + bias[i]);
                num_success++;
                num_failed = 0;
            } else if (evals < maxevals) {
                // Try opposite direction
                for (int i = 0; i < dim; i++) {
                    newsol[i] = static_cast<tDomain>(
                        static_cast<double>(sol[i]) - dif[i] - bias[i]);
                }
                problem.fix(newsol);
                newfit = problem.fitness(newsol);
                evals++;

                if (newfit > fitness) {
                    sol     = newsol;
                    fitness = newfit;
                    // decrement_bias: reinforce opposite direction
                    for (int i = 0; i < dim; i++)
                        bias[i] = bias[i] - 0.4 * (dif[i] + bias[i]);
                    num_success++;
                    num_failed = 0;
                } else {
                    for (int i = 0; i < dim; i++)
                        bias[i] /= 2.0;
                    num_success = 0;
                    num_failed++;
                }
            }

            if      (num_success >= 5) { num_success = 0; delta *= 2.0; }
            else if (num_failed  >= 3) { num_failed  = 0; delta /= 2.0; }
        }

        return ResultMH<tDomain>(sol, fitness, static_cast<unsigned>(evals));
    }
};
