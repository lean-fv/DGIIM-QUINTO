#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <cmath>

using namespace std;

// Overload << for vector
template <typename S>
ostream &operator<<(ostream &os, const vector<S> &vector) {

  // Printing all the elements using <<
  for (auto i : vector)
    os << i << " ";
  return os;
}

/**
 * @brief Information about a dataset pair used for experiments.
 *
 * Holds the filenames for training and testing CSVs and the minimum and
 * maximum per-asset weights used when building the ProblemPortfolio.
 */
    struct DatasetInfo {
        string train_file;
        string test_file;
        float min_w;
        float max_w;
    };


/**
 * @brief Application configuration container.
 *
 * This struct holds simple key/value parameters used to configure the
 * execution (stored in `params`) and a list of datasets to process
 * (stored in `datasets`).
 *
 * - `params`: generic configuration options loaded from the
 *   configuration file (e.g. `seed`, `max_evals`, `num_runs`,
 *   `data_folder`, etc.).
 * - `datasets`: a vector of `DatasetInfo` objects describing train/test
 *   CSV filenames and the per-asset weight bounds to use for each
 *   dataset.
 */
struct AppConfig {
    map<string, string> params;
    vector<DatasetInfo> datasets;
};


/**
 * @brief Read a configuration file into an AppConfig object.
 *
 * The function opens `filename` and parses it line-by-line. Lines beginning
 * with `#` or empty lines are ignored. Each non-comment line is expected to
 * contain a single key/value pair separated by an `=` character. 
 *
 * If the key is "dataset", the value is expected to be a comma-separated 
 * string (train_file, test_file, min_w, max_w) which is parsed and added 
 * to `config.datasets`. Any other key/value pairs are stored directly in 
 * `config.params` as strings. If the file cannot be opened, a warning is 
 * printed and an empty/default `AppConfig` is returned.
 *
 * @param filename Path to the configuration file.
 * @return AppConfig container populated with parsed `params` and `datasets`.
 */
AppConfig readConfigFile(const string& filename) {
    AppConfig config;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Warning: Could not open file '" << filename 
             << "'. Default parameters will be used." << endl;
    } else {
        string line;
        while (getline(file, line)) {
            // Process only lines that are not empty and do not start with '#'
            if (!line.empty() && line[0] != '#') {
                
                size_t delimiterPos = line.find('=');
                
                // If the delimiter '=' is found, extract key and value
                if (delimiterPos != string::npos) {
                    string key = line.substr(0, delimiterPos);
                    string value = line.substr(delimiterPos + 1);
                    
                    // Check if the key corresponds to a dataset definition
                    if (key == "dataset") {
                        vector<string> tokens;
                        string token;
                        istringstream tokenStream(value);
                        
                        // Split the value string by commas
                        while (getline(tokenStream, token, ',')) {
                            tokens.push_back(token);
                        }
                        
                        // A valid dataset must have exactly 4 parameters
                        if (tokens.size() == 4) {
                            DatasetInfo ds;
                            ds.train_file = tokens[0];
                            ds.test_file = tokens[1];
                            ds.min_w = stof(tokens[2]); // Convert to float
                            ds.max_w = stof(tokens[3]); // Convert to float
                            config.datasets.push_back(ds);
                        } else {
                            cerr << "Warning: Incorrect dataset format -> " << value << endl;
                        }
                    } 
                    // For any other parameter, store it in the map
                    else {
                        config.params[key] = value;
                    }
                }
            }
        }
        file.close();
    }
    
    return config;
}

/**
 * @brief Parse a CSV file and extract numeric asset returns.
 *
 * Reads the CSV file at `csvpath`. The function expects a header line and
 * assumes the first column is a non-numeric identifier (e.g., a date), so
 * it is skipped. Each subsequent column is parsed as a float return for an
 * asset. Missing numeric entries are treated as 0.0.
 *
 * @param csvpath Path to the CSV file to parse.
 * @return vector<vector<float>> Parsed returns matrix where each inner
 *         vector corresponds to a period (row) and contains one float per asset.
 *         Returns an empty vector on file-open or format errors.
 */
vector<vector<float>> parseCSV(const string &csvpath) {
    vector<vector<float>> historical_data;
    ifstream f(csvpath);

    // 1. Check if the file was opened successfully
    if (f) {
        string line, cell;
        
        // 2. Check if we can read the first line (header)
        if (getline(f, line)) {
            stringstream sshead(line);
            vector<string> headers;
            
            while (getline(sshead, cell, ',')) {
                headers.push_back(cell);
            }
            
            // 3. Check if there are at least 2 columns (date + 1 asset)
            if (headers.size() >= 2) {
                size_t ncols = headers.size();
                size_t num_assets = ncols - 1; // Assume the first column is the date

                // Process the remaining lines
                while (getline(f, line)) {
                    if (!line.empty()) {
                        stringstream ss(line);
                        
                        // Skip the first column (date)
                        getline(ss, cell, ',');
                        
                        vector<float> daily_returns;
                        for (size_t i = 0; i < num_assets; ++i) {
                            if (!getline(ss, cell, ',')) {
                                cell = "0"; // Default value if a data point is missing
                            }
                            daily_returns.push_back(stof(cell));
                        }
                        historical_data.push_back(std::move(daily_returns));
                    }
                }
            }
        }
    }

    return historical_data;
}

/**
 * @brief Compute summary statistics for a collection of runs.
 *
 * Given vectors with per-run values this function computes summary
 * statistics: best (max), worst (min), mean and population standard
 * deviation for the fitness values in `data`. It also computes the mean
 * of the provided `returns`, the average number of evaluations from
 * `evals`, and the mean execution time from `times`.
 *
 * @param data Vector of fitness values (one element per run).
 * @param returns Vector of return/benefit values corresponding to each run.
 * @param evals Vector with the number of evaluations used in each run.
 * @param times Vector of execution times in seconds for each run.
 * @param[out] best Output parameter set to the maximum value in `data`.
 * @param[out] worst Output parameter set to the minimum value in `data`.
 * @param[out] mean Output parameter set to the arithmetic mean of `data`.
 * @param[out] std_dev Output parameter set to the population standard deviation of `data`.
 * @param[out] mean_return Output parameter set to the arithmetic mean of `returns`.
 * @param[out] mean_evals Output parameter set to the arithmetic mean of `evals`.
 * @param[out] mean_time Output parameter set to the arithmetic mean of `times` (seconds).
 */
/**
 * @brief Parse a comma-separated string of integers into a vector.
 *
 * Splits @p s on commas and converts each token to an integer with stoi().
 * Tokens that cannot be converted are silently skipped.
 * Leading/trailing whitespace inside tokens is not stripped; callers should
 * ensure the string is well-formed (e.g. "1,5,10,30").
 *
 * @param s Comma-separated string of integer tokens (e.g. "1,2,3,4,5").
 * @return  Vector of parsed integers in the same order as the input tokens.
 *          Returns an empty vector if @p s is empty or contains no valid tokens.
 */
vector<int> parseIntList(const string& s) {
    vector<int> result;
    if (s.empty()) return result;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        try { result.push_back(stoi(token)); }
        catch (...) {}
    }
    return result;
}

void calculateStatistics(const vector<float>& data, 
                         const vector<float>& returns, 
                         const vector<int>& evals, 
                         const vector<double>& times, 
                         float& best, float& worst, float& mean, float& std_dev, 
                         float& mean_return, float& mean_evals, double& mean_time) {
    
    if (!data.empty()){
        // 1. Fitness (Best, Worst, Mean, Std Dev)
        best = data[0];
        worst = data[0];
        float sum = 0.0f;
        
        for (float val : data) {
            if (val > best) best = val;
            if (val < worst) worst = val;
            sum += val;
        }
        
        mean = sum / data.size();
        
        float variance_sum = 0.0f;
        for (float val : data) {
            variance_sum += (val - mean) * (val - mean);
        }
        std_dev = sqrt(variance_sum / data.size());

        // 2. Return (Beneficio)
        mean_return = 0.0f;
        if (!returns.empty()) {
            float return_sum = 0.0f;
            for (float r : returns) {
                return_sum += r;
            }
            mean_return = return_sum / returns.size();
        }

        // 3. Evaluations
        mean_evals = 0.0f;
        if (!evals.empty()) {
            float eval_sum = 0.0f;
            for (int e : evals) {
                eval_sum += e;
            }
            mean_evals = eval_sum / evals.size();
        }

        // 4. Times
        mean_time = 0.0;
        if (!times.empty()) {
            double time_sum = 0.0;
            for (double t : times) {
                time_sum += t;
            }
            mean_time = time_sum / times.size();
        }
    }
    else {
        best = 0.0f;
        worst = 0.0f;
        mean = 0.0f;
        std_dev = 0.0f;
        mean_return = 0.0f;
        mean_evals = 0.0f;
        mean_time = 0.0;
    }
}
