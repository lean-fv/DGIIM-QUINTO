#pragma once

// CEC2017 C library must be linked separately (cec17.c + cec17_test_func.c).
extern "C" {
#include "cec17.h"
}

#include "problem.h"
#include <vector>
#include <algorithm>

using namespace std;

/**
 * ProblemCEC2017
 *
 * Wraps the CEC2017 Single-Objective Bound-Constrained benchmark suite
 * (functions F1-F29) as a maximisation problem so it fits the existing
 * MH<double> interface. The search domain is [-bound, bound]^D; the
 * standard CEC2017 bound is 100.0 but it is configurable at construction.
 *
 * The constructor calls cec17_init() internally, so the C library global
 * state is always initialised before any fitness evaluation can occur.
 * Callers do not need to call cec17_init() separately.
 *
 * The raw CEC2017 fitness is negated so that the framework's maximisation
 * correctly minimises the underlying benchmark function.
 */
class ProblemCEC2017 : public Problem<double> {
private:
    int    dim;
    double bound; // Domain is [-bound, bound] for every variable

    mutable vector<double> eval_buf; // Reusable buffer for cec17_fitness calls

public:
    /**
     * @brief Construct a ProblemCEC2017 and initialise the CEC2017 C library.
     *
     * Calls cec17_init() so the C library global state is ready before any
     * call to fitness(). Also pre-allocates the internal evaluation buffer
     * to avoid repeated heap allocation during optimisation.
     *
     * @param algname   Algorithm name forwarded to cec17_init for output file naming.
     * @param funcid    Benchmark function index (1-29).
     * @param dimension Number of decision variables (D).
     * @param bound     Half-width of the search domain; domain is [-bound, bound] (default 100.0).
     */
    ProblemCEC2017(const string& algname, int funcid, int dimension, double bound = 100.0)
        : dim(dimension), bound(bound), eval_buf(dimension) {
        cec17_init(algname.c_str(), funcid, dimension);
    }

    virtual ~ProblemCEC2017() {}

    /**
     * @brief Return the number of decision variables.
     *
     * @return size_t Dimensionality D of the search space.
     */
    size_t getSolutionSize() override {
        return static_cast<size_t>(dim);
    }

    /**
     * @brief Return the allowed range for each decision variable.
     *
     * @return pair<double, double> Domain bounds {-bound, bound}.
     */
    pair<double, double> getSolutionDomainRange() override {
        return {-bound, bound};
    }

    /**
     * @brief Evaluate the fitness of a solution.
     *
     * Copies the solution into a reusable internal buffer (required because
     * the CEC2017 C API takes a non-const pointer) and returns the negated
     * raw CEC2017 value so that higher fitness corresponds to a better
     * (lower) benchmark function value.
     *
     * @param solution Candidate solution vector of length dim.
     * @return tFitness Negated CEC2017 raw value; higher is better.
     */
    tFitness fitness(const tSolution<double>& solution) override {
        eval_buf.assign(solution.begin(), solution.end());
        double raw = cec17_fitness(eval_buf.data());
        return -static_cast<tFitness>(raw);
    }

    /**
     * @brief Create a uniformly random feasible solution.
     *
     * Each variable is sampled independently from U[-bound, bound].
     *
     * @return tSolution<double> A random solution vector of length dim.
     */
    tSolution<double> createSolution() override {
        tSolution<double> sol(dim);
        for (auto& v : sol)
            v = Random::get<double>(-bound, bound);
        return sol;
    }

    /**
     * @brief Check whether a solution lies within the search domain.
     *
     * A solution is valid if every variable satisfies -bound <= v <= bound.
     *
     * @param solution Candidate solution vector to check.
     * @return true if all variables are within [-bound, bound], false otherwise.
     */
    bool isValid(const tSolution<double>& solution) override {
        bool valid = true;
        for (size_t i = 0; i < solution.size() && valid; i++)
            valid = (solution[i] >= -bound && solution[i] <= bound);
        return valid;
    }

    /**
     * @brief Clamp each variable to the search domain [-bound, bound].
     *
     * Modifies the solution in-place so that every variable satisfies
     * the bound constraints required by the CEC2017 benchmark.
     *
     * @param solution Solution vector to be repaired in-place.
     */
    void fix(tSolution<double>& solution) override {
        for (auto& v : solution)
            v = clamp(v, -bound, bound);
    }
};
