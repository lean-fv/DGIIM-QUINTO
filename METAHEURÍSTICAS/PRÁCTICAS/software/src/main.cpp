#include <iomanip>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <sstream>

// Utilities and data structures
#include "util.h"

// Real problem classes
#include "portfolio.h"
#include "cec2017problem.h"

// All algorithms
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "geneticalgorithm.h"
#include "generational_ga.h"
#include "steadystate_ga.h"
#include "memetic.h"
#include "simulatedannealing.h"
#include "bmb.h"
#include "ils.h"
#include "bbo.h"
#include "bbo_local_search.h"
#include "bbo_improved.h"
#include "bbo_solis_wets.h"

using namespace std;

/**
 * Entry point supporting two operation modes:
 *
 *  mode=portfolio (default)
 *    Portfolio optimisation experiment identical to the original behaviour.
 *
 *  mode=cec2017
 *    Runs BBO (and RandomSearch as baseline) on the CEC2017 single-objective
 *    benchmark suite.  Results are written as CSV files to ./results/ and a
 *    summary table is printed for each (function, dimension) pair.
 *    Config keys: cec_funcs (comma-separated function IDs, default 1-10),
 *                 cec_dims  (comma-separated dimensions,   default 10).
 *
 * @param argc Number of command-line arguments.
 * @param argv argv[1], if present, is the path to the configuration file.
 * @return 0 on success.
 */
int main(int argc, char *argv[]) {
    // 1. Read configuration
    string config_filename = (argc > 1) ? argv[1] : "";
    AppConfig config = readConfigFile(config_filename);

    // 2. Extract simple parameters
    long int seed    = config.params.count("seed")      ? stol(config.params["seed"])      : 42;
    int max_evals    = config.params.count("max_evals") ? stoi(config.params["max_evals"]) : 10000;
    int num_runs     = config.params.count("num_runs")  ? stoi(config.params["num_runs"])  : 50;
    float lambda     = config.params.count("lambda")    ? stof(config.params["lambda"])    : 500.0f;
    float ls_ratio   = config.params.count("ls_ratio")  ? stof(config.params["ls_ratio"])  : 0.4f;

    // GA / MA parameters
    int   ga_pop_size  = config.params.count("ga_pop_size")  ? stoi(config.params["ga_pop_size"])  : 100;
    float agg_pc       = config.params.count("agg_pc")       ? stof(config.params["agg_pc"])       : 0.8f;
    float ga_pm        = config.params.count("ga_pm")        ? stof(config.params["ga_pm"])        : 0.1f;
    int   ga_tourney_k = config.params.count("ga_tourney_k") ? stoi(config.params["ga_tourney_k"]) : 3;

    // MA-specific parameters
    float ma_bl_ratio    = config.params.count("ma_bl_ratio")    ? stof(config.params["ma_bl_ratio"])    : 0.15f;
    int   ma_bl_maxevals = config.params.count("ma_bl_maxevals") ? stoi(config.params["ma_bl_maxevals"]) : 100;
    int   ma_gen_period  = config.params.count("ma_gen_period")  ? stoi(config.params["ma_gen_period"])  : 10;
    float ma_rand_prob   = config.params.count("ma_rand_prob")   ? stof(config.params["ma_rand_prob"])   : 0.1f;
    float ma_top_ratio   = config.params.count("ma_top_ratio")   ? stof(config.params["ma_top_ratio"])   : 0.1f;

    // ES (Simulated Annealing) parameters
    float es_mu  = config.params.count("es_mu")  ? stof(config.params["es_mu"])  : 0.2f;
    float es_phi = config.params.count("es_phi") ? stof(config.params["es_phi"]) : 0.3f;
    float es_tf  = config.params.count("es_tf")  ? stof(config.params["es_tf"])  : 1e-3f;

    // BMB parameters
    int bmb_restarts          = config.params.count("bmb_restarts")          ? stoi(config.params["bmb_restarts"])          : 5;
    int bmb_evals_per_restart = config.params.count("bmb_evals_per_restart") ? stoi(config.params["bmb_evals_per_restart"]) : 2000;

    // ILS parameters
    float ils_mut_ratio      = config.params.count("ils_mut_ratio")      ? stof(config.params["ils_mut_ratio"])      : 0.20f;
    int   ils_num_iters      = config.params.count("ils_num_iters")      ? stoi(config.params["ils_num_iters"])      : 5;
    int   ils_evals_per_iter = config.params.count("ils_evals_per_iter") ? stoi(config.params["ils_evals_per_iter"]) : 2000;

    // BBO / BBOImproved parameters (shared)
    int    bbo_elitism   = config.params.count("bbo_elitism")   ? stoi(config.params["bbo_elitism"])   : 2;
    double bbo_imp_sigma = config.params.count("bbo_imp_sigma") ? stod(config.params["bbo_imp_sigma"]) : 0.1;

    // BBOLocalSearch parameters (portfolio only)
    float bbo_ls_ratio    = config.params.count("bbo_ls_ratio")    ? stof(config.params["bbo_ls_ratio"])    : ls_ratio;
    int   bbo_ls_every    = config.params.count("bbo_ls_every")    ? stoi(config.params["bbo_ls_every"])    : 5;
    int   bbo_ls_maxevals = config.params.count("bbo_ls_maxevals") ? stoi(config.params["bbo_ls_maxevals"]) : 200;

    // BBO parameters for CEC2017 (independent of portfolio GA params)
    int    bbo_pop_size = config.params.count("bbo_pop_size") ? stoi(config.params["bbo_pop_size"]) : 50;
    float  bbo_mutation = config.params.count("bbo_mutation") ? stof(config.params["bbo_mutation"]) : 0.01f;
    double cec_bound    = config.params.count("cec_bound")    ? stod(config.params["cec_bound"])    : 100.0;

    // BBOSolisWets parameters (portfolio only)
    double sw_delta     = config.params.count("sw_delta")     ? stod(config.params["sw_delta"])     : 0.2;
    int    sw_every     = config.params.count("sw_every")     ? stoi(config.params["sw_every"])     : 5;
    int    sw_max_evals = config.params.count("sw_max_evals") ? stoi(config.params["sw_max_evals"]) : 200;

    // Mode and dataset folder
    string mode        = config.params.count("mode")        ? config.params["mode"]        : "portfolio";
    string data_folder = config.params.count("data_folder") ? config.params["data_folder"] : "datos_portfolio_2526/";

    // CEC2017-specific parameters
    string cec_funcs_str = config.params.count("cec_funcs") ? config.params["cec_funcs"] : "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30";
    string cec_dims_str  = config.params.count("cec_dims")  ? config.params["cec_dims"]  : "10,30,50";
    int    cec_runs      = config.params.count("cec_runs")  ? stoi(config.params["cec_runs"])  : 10;

    vector<int> cec_func_ids  = parseIntList(cec_funcs_str);
    vector<int> cec_dimensions = parseIntList(cec_dims_str);

    // 3. Print configuration — only show parameters relevant to the active mode
    cout << "========================================" << endl;
    cout << "          EXECUTION PARAMETERS          " << endl;
    cout << "========================================" << endl;
    cout << "Config file    : " << config_filename << endl;
    cout << "Mode           : " << mode << endl;
    cout << "Seed           : " << seed << endl;
    cout << "Num Runs       : " << num_runs << endl;

    if (mode == "portfolio") {
        cout << "Max Evals      : " << max_evals << endl;
        cout << "Lambda         : " << lambda << endl;
        cout << "LS Ratio       : " << ls_ratio << endl;
        cout << "GA Pop Size    : " << ga_pop_size << endl;
        cout << "AGG Pc         : " << agg_pc << endl;
        cout << "GA Pm          : " << ga_pm << endl;
        cout << "GA Tourney k   : " << ga_tourney_k << endl;
        cout << "MA BL Ratio    : " << ma_bl_ratio << endl;
        cout << "MA BL MaxEvals : " << ma_bl_maxevals << endl;
        cout << "MA Gen Period  : " << ma_gen_period << endl;
        cout << "MA Rand Prob   : " << ma_rand_prob << endl;
        cout << "MA Top Ratio   : " << ma_top_ratio << endl;
        cout << "ES Mu          : " << es_mu << endl;
        cout << "ES Phi         : " << es_phi << endl;
        cout << "ES Tf          : " << es_tf << endl;
        cout << "BMB Restarts   : " << bmb_restarts << endl;
        cout << "BMB Evals/Rest : " << bmb_evals_per_restart << endl;
        cout << "ILS Mut Ratio  : " << ils_mut_ratio << endl;
        cout << "ILS Num Iters  : " << ils_num_iters << endl;
        cout << "ILS Evals/Iter : " << ils_evals_per_iter << endl;
        cout << "BBO Elitism    : " << bbo_elitism << endl;
        cout << "BBO Imp Sigma  : " << bbo_imp_sigma << endl;
        cout << "BBO LS Ratio   : " << bbo_ls_ratio << endl;
        cout << "BBO LS Every   : " << bbo_ls_every << " gens" << endl;
        cout << "BBO LS MaxEvals: " << bbo_ls_maxevals << endl;
        cout << "SW Delta       : " << sw_delta      << endl;
        cout << "SW Every       : " << sw_every      << " gens" << endl;
        cout << "SW MaxEvals    : " << sw_max_evals  << endl;
        cout << "Data Folder    : " << data_folder << endl;
        cout << "Loaded datasets: " << config.datasets.size() << endl;
    } else if (mode == "cec2017") {
        cout << "BBO Pop Size   : " << bbo_pop_size  << endl;
        cout << "BBO Mutation   : " << bbo_mutation  << endl;
        cout << "BBO Elitism    : " << bbo_elitism   << endl;
        cout << "BBO Imp Sigma  : " << bbo_imp_sigma << endl;
        cout << "SW Delta       : " << sw_delta      << endl;
        cout << "SW Every       : " << sw_every      << " gens" << endl;
        cout << "SW MaxEvals    : " << sw_max_evals  << endl;
        cout << "CEC Bound      : " << cec_bound     << endl;
        cout << "CEC Funcs      : " << cec_funcs_str << endl;
        cout << "CEC Dims       : " << cec_dims_str  << endl;
        cout << "CEC Runs       : " << cec_runs      << endl;
    }
    cout << "========================================\n" << endl;

    // =========================================================================
    // RESULTS DIRECTORY
    // =========================================================================
    string results_folder = "./results/";
    if (!filesystem::exists(results_folder)) {
        filesystem::create_directory(results_folder);
        cout << "Directory " << results_folder << " created successfully." << endl;
    }

    // =========================================================================
    // MODE: PORTFOLIO
    // =========================================================================
    if (mode == "portfolio") {
        // Extract dataset list
        vector<DatasetInfo> datasets = config.datasets;
        if (datasets.empty()) {
            cout << "Warning: No datasets found in config. Loading defaults..." << endl;
            datasets = {
                {"ibex_35.csv",  "ibex_35_2025.csv",  0.005f, 0.08f},
                {"syp_100.csv",  "syp_100_2025.csv",  0.005f, 0.05f},
                {"syp_500.csv",  "syp_500_2025.csv",  0.005f, 0.02f}
            };
        }

        // Create portfolio algorithms
        RandomSearch<float>   ralg;
        GreedySearch          rgreedy;
        LocalSearch<float>    rlocal(ls_ratio);
        GenerationalGA<float> agg_arit(ga_pop_size, agg_pc, ga_pm, CrossoverType::ARITHMETIC, ga_tourney_k);
        GenerationalGA<float> agg_blx (ga_pop_size, agg_pc, ga_pm, CrossoverType::BLX_ALPHA,  ga_tourney_k);
        SteadyStateGA<float>  age_arit(ga_pop_size, ga_pm, CrossoverType::ARITHMETIC, ga_tourney_k);
        SteadyStateGA<float>  age_blx (ga_pop_size, ga_pm, CrossoverType::BLX_ALPHA,  ga_tourney_k);
        AM_All<float>         am_all(ga_pop_size, agg_pc, ga_pm, CrossoverType::BLX_ALPHA,
                                     ga_tourney_k, ma_bl_ratio, ma_bl_maxevals, ma_gen_period);
        AM_Rand<float>        am_rand(ga_pop_size, agg_pc, ga_pm, CrossoverType::BLX_ALPHA,
                                      ga_tourney_k, ma_bl_ratio, ma_bl_maxevals, ma_gen_period, ma_rand_prob);
        AM_Best<float>        am_best(ga_pop_size, agg_pc, ga_pm, CrossoverType::BLX_ALPHA,
                                      ga_tourney_k, ma_bl_ratio, ma_bl_maxevals,
                                      ma_gen_period, ma_top_ratio);
        SimulatedAnnealing<float> es_alg(ls_ratio, es_mu, es_phi, es_tf);
        BMB<float>                bmb_alg(ls_ratio, bmb_restarts, bmb_evals_per_restart);
        ILS_BL<float>             ils_bl_alg(ls_ratio, ils_mut_ratio, ils_num_iters, ils_evals_per_iter);
        ILS_ES<float>             ils_es_alg(ls_ratio, es_mu, es_phi, es_tf,
                                              ils_mut_ratio, ils_num_iters, ils_evals_per_iter);
        BBO<float>             bbo_alg(ga_pop_size, ga_pm);
        BBOLocalSearch<float>  bbo_ls_alg(ga_pop_size, ga_pm, bbo_ls_ratio, bbo_ls_every, bbo_ls_maxevals);
        BBOImproved<float>     bbo_imp_alg(ga_pop_size, ga_pm, bbo_elitism, bbo_imp_sigma);
        BBOSolisWets<float>    bbo_sw_alg(ga_pop_size, ga_pm, sw_delta, sw_every, sw_max_evals);

        vector<pair<string, MH<float>*>> algorithms = {
            {"RandomSearch",   &ralg},
            {"Greedy",         &rgreedy},
            {"LocalSearch",    &rlocal},
            {"AGG_Arit",        &agg_arit},
            {"AGG_BLX",         &agg_blx},
            {"AGE_Arit",        &age_arit},
            {"AGE_BLX",         &age_blx},
            {"AM_All",          &am_all},
            {"AM_Rand",        &am_rand},
            {"AM_Best",        &am_best},
            {"ES",             &es_alg},
            {"BMB",            &bmb_alg},
            {"ILS_BL",         &ils_bl_alg},
            {"ILS_ES",         &ils_es_alg},
            {"BBO",            &bbo_alg},
            {"BBOLocalSearch", &bbo_ls_alg},
            {"BBOImproved",    &bbo_imp_alg},
            {"BBOSolisWets",   &bbo_sw_alg},
        };

        // Dataset loop
        for (const auto& dataset : datasets) {
            vector<vector<float>> train_data = parseCSV(data_folder + dataset.train_file);
            vector<vector<float>> test_data  = parseCSV(data_folder + dataset.test_file);

            if (!train_data.empty() && !test_data.empty()) {
                ProblemPortfolio prob_train(train_data, dataset.min_w, dataset.max_w, lambda);
                ProblemPortfolio prob_test (test_data,  dataset.min_w, dataset.max_w, lambda);
                Problem<float>* problem_ptr = dynamic_cast<Problem<float>*>(&prob_train);

                cout << "\n=================================================================================" << endl;
                cout << "Processing: " << dataset.train_file
                     << " | Test: "    << dataset.test_file << endl;
                cout << "=================================================================================" << endl;

                string base_name         = dataset.train_file.substr(0, dataset.train_file.find_last_of('.'));
                string out_train_filename = results_folder + base_name + "_train.csv";
                string out_test_filename  = results_folder + base_name + "_test.csv";

                ofstream out_train(out_train_filename);
                ofstream out_test (out_test_filename);
                out_train << "run,alg,fitness,evaluations\n";
                out_test  << "run,alg,fitness,evaluations\n";

                map<string, vector<float>>  train_results, test_results, test_returns;
                map<string, vector<int>>    eval_counts;
                map<string, vector<double>> exec_times;

                for (int run = 0; run < num_runs; ++run) {
                    cout << "  -> Run " << run + 1 << "/" << num_runs << "...\r" << flush;

                    for (auto& [alg_name, mh] : algorithms) {
                        Random::seed(seed + run);

                        auto t0 = chrono::high_resolution_clock::now();
                        ResultMH<float> result = mh->optimize(*problem_ptr, max_evals);
                        auto t1 = chrono::high_resolution_clock::now();
                        double exec_sec = chrono::duration<double>(t1 - t0).count();

                        tFitness fitness_test  = prob_test.fitness(result.solution);
                        float    return_test   = prob_test.getReturn(result.solution)
                                                 * static_cast<float>(test_data.size());

                        out_train << (run + 1) << "," << alg_name << ","
                                  << result.fitness    << "," << result.evaluations << "\n";
                        out_test  << (run + 1) << "," << alg_name << ","
                                  << fitness_test      << "," << result.evaluations << "\n";

                        train_results[alg_name].push_back(result.fitness);
                        test_results [alg_name].push_back(fitness_test);
                        test_returns [alg_name].push_back(return_test);
                        eval_counts  [alg_name].push_back(result.evaluations);
                        exec_times   [alg_name].push_back(exec_sec);
                    }
                }

                cout << "  -> Finished all runs!" << endl;
                out_train.close();
                out_test.close();

                // Summary table
                cout << "\nRESULTS TABLE (" << num_runs << " runs):" << endl;
                cout << left  << setw(15) << "Algorithm"
                     << right << setw(12) << "Best(Tr)" << setw(12) << "Worst(Tr)"
                     << setw(12) << "Mean(Tr)"  << setw(12) << "Std(Tr)"
                     << setw(12) << "Best(Ts)"  << setw(12) << "Worst(Ts)"
                     << setw(12) << "Mean(Ts)"  << setw(12) << "Std(Ts)"
                     << setw(15) << "Return(Ts)"
                     << setw(12) << "Evals"     << setw(12) << "Time(s)" << endl;
                cout << string(148, '-') << endl;

                for (auto& [alg_name, mh] : algorithms) {
                    float  best_tr, worst_tr, mean_tr, std_tr, ret_tr, evals_tr;
                    float  best_ts, worst_ts, mean_ts, std_ts, ret_ts, evals_ts;
                    double time_tr, time_ts;

                    calculateStatistics(train_results[alg_name], {}, eval_counts[alg_name],
                                        exec_times[alg_name],
                                        best_tr, worst_tr, mean_tr, std_tr, ret_tr, evals_tr, time_tr);
                    calculateStatistics(test_results[alg_name], test_returns[alg_name], {}, {},
                                        best_ts, worst_ts, mean_ts, std_ts, ret_ts, evals_ts, time_ts);

                    cout << left  << setw(15) << alg_name
                         << right << fixed    << setprecision(3)
                         << setw(12) << best_tr  << setw(12) << worst_tr
                         << setw(12) << mean_tr  << setw(12) << std_tr
                         << setw(12) << best_ts  << setw(12) << worst_ts
                         << setw(12) << mean_ts  << setw(12) << std_ts
                         << setw(15) << ret_ts
                         << setw(12) << evals_tr << setw(12) << time_tr << endl;
                }
            } else {
                cerr << "Error: Could not load data for " << dataset.train_file
                     << " / " << dataset.test_file << ". Skipping." << endl;
            }
        }

    // =========================================================================
    // MODE: CEC2017
    // =========================================================================
    } else if (mode == "cec2017") {

        if (cec_func_ids.empty())  { cerr << "No CEC functions specified.\n"; return 1; }
        if (cec_dimensions.empty()){ cerr << "No CEC dimensions specified.\n"; return 1; }

        // BBO algorithms for double domain (CEC2017 uses its own pop/mutation params)
        BBOImproved<double>  bbo_imp_d(bbo_pop_size, bbo_mutation, bbo_elitism, bbo_imp_sigma);

        vector<pair<string, MH<double>*>> algs_d = {
            {"BBOImproved",  &bbo_imp_d},
        };

        // Create CEC17 output directories nested under results_cec17/.
        string cec_base = "results_cec17";
        filesystem::create_directories(cec_base);
        cec17_set_base_dir(cec_base.c_str());
        for (auto& [alg_name, mh] : algs_d) {
            string dir = cec_base + "/results_" + alg_name;
            if (!filesystem::exists(dir))
                filesystem::create_directories(dir);
        }

        // Valid CEC17 dimensions; filter out unsupported values.
        static const int valid_dims[] = {2, 10, 20, 30, 50, 100};
        cec_dimensions.erase(
            remove_if(cec_dimensions.begin(), cec_dimensions.end(), [](int d) {
                for (int v : valid_dims) if (v == d) return false;
                cerr << "Warning: dimension " << d << " not supported by CEC17; skipping.\n";
                return true;
            }),
            cec_dimensions.end());

        for (int dim : cec_dimensions) {
            int max_evals_cec = max_evals * dim;

            for (int funcid : cec_func_ids) {
                cout << "\n========================================================" << endl;
                cout << "  F" << funcid << "  D=" << dim
                     << "  max_evals=" << max_evals_cec
                     << "  runs=" << cec_runs << endl;
                cout << "========================================================" << endl;

                // Per-algorithm error collectors
                map<string, vector<double>> errors;
                map<string, vector<double>> exec_times;

                for (int run = 0; run < cec_runs; ++run) {
                    cout << "  -> Run " << run + 1 << "/" << cec_runs << "...\r" << flush;

                    for (auto& [alg_name, mh] : algs_d) {
                        // ProblemCEC2017 calls cec17_init internally, resetting
                        // the evaluation counter and output file for this run.
                        ProblemCEC2017 prob(alg_name, funcid, dim, cec_bound);
                        Random::seed(seed + run);

                        auto t0 = chrono::high_resolution_clock::now();
                        ResultMH<double> result = mh->optimize(prob, max_evals_cec);
                        auto t1 = chrono::high_resolution_clock::now();
                        double exec_sec = chrono::duration<double>(t1 - t0).count();

                        // Undo the negation to recover the raw CEC17 fitness
                        double raw_fit = -static_cast<double>(result.fitness);
                        double error   = cec17_error(raw_fit);

                        errors    [alg_name].push_back(error);
                        exec_times[alg_name].push_back(exec_sec);
                    }
                }

                cout << "  -> Done." << endl;

                // Print summary table for this (func, dim)
                cout << "\n  SUMMARY  F" << funcid << "  D=" << dim << endl;
                cout << "  " << left  << setw(15) << "Algorithm"
                     << right << setw(14) << "Mean Error"
                     << setw(14) << "Std Error"
                     << setw(14) << "Best Error"
                     << setw(14) << "Worst Error"
                     << setw(12) << "Time(s)" << endl;
                cout << "  " << string(83, '-') << endl;

                for (auto& [alg_name, mh] : algs_d) {
                    auto& errs = errors[alg_name];
                    auto& times = exec_times[alg_name];
                    int n = static_cast<int>(errs.size());

                    double sum = 0, sumsq = 0, best = errs[0], worst = errs[0], tsum = 0;
                    for (int r = 0; r < n; r++) {
                        sum   += errs[r];
                        sumsq += errs[r] * errs[r];
                        if (errs[r] < best)  best  = errs[r];
                        if (errs[r] > worst) worst = errs[r];
                        tsum  += times[r];
                    }
                    double mean   = sum / n;
                    double stddev = sqrt(sumsq / n - mean * mean);

                    cout << "  " << left  << setw(15) << alg_name
                         << right << scientific << setprecision(4)
                         << setw(14) << mean  << setw(14) << stddev
                         << setw(14) << best  << setw(14) << worst
                         << fixed    << setprecision(3)
                         << setw(12) << tsum / n << endl;
                }
            }
        }

    } else {
        cerr << "Unknown mode '" << mode << "'. Use mode=portfolio or mode=cec2017." << endl;
        return 1;
    }

    return 0;
}
