/**
 * @file benchmark_eigen.cpp
 * @brief Production-grade Latency & Throughput Benchmark for NIDS using Eigen 3.
 *
 * Compares single-sample real-time packet-window inference latency and
 * multi-core throughput on realistic network traffic using standard,
 * industry-proven Eigen 3 routines with AVX2 SIMD acceleration.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <omp.h>

#include "nids_engine.hpp"

struct BenchmarkSample {
    std::vector<int> indices;
    int true_label;
};

// Loads SVMLight samples into RAM to isolate pure computation latency from disk I/O
std::vector<BenchmarkSample> load_dataset(const std::string& filepath, int max_samples) {
    std::vector<BenchmarkSample> samples;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[Benchmark] Error: Could not open dataset: " << filepath << std::endl;
        return samples;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        BenchmarkSample sample;
        if (!(iss >> sample.true_label)) continue;

        std::string token;
        while (iss >> token) {
            size_t colon_pos = token.find(':');
            if (colon_pos != std::string::npos) {
                int idx = std::stoi(token.substr(0, colon_pos));
                if (idx >= 0 && idx < NidsEngine::INPUT_DIM) {
                    sample.indices.push_back(idx);
                }
            }
        }
        samples.push_back(std::move(sample));
        if (max_samples > 0 && static_cast<int>(samples.size()) >= max_samples) {
            break;
        }
    }
    return samples;
}

int main(int argc, char* argv[]) {
    std::string weights_path = "configs/weights_v5.bin";
    std::string test_path = "datasets/CICIDS2017/friday/friday_v5_flow_test.txt";
    int max_samples = 50000;

    if (argc >= 2) test_path = argv[1];
    if (argc >= 3) max_samples = std::stoi(argv[2]);
    if (argc >= 4) weights_path = argv[3];

    std::cout << "======================================================================" << std::endl;
    std::cout << "       NIDS INFERENCE BENCHMARK (EIGEN 3 + AVX2 / FMA ENGINE)         " << std::endl;
    std::cout << "======================================================================" << std::endl;

    NidsEngine engine;
    std::cout << "[1/4] Loading weights from: " << weights_path << " ... ";
    if (!engine.load(weights_path)) {
        std::cerr << "Failed to load weights." << std::endl;
        return 1;
    }
    std::cout << "[OK] (Topology: " << NidsEngine::INPUT_DIM << " -> "
              << NidsEngine::H0_DIM << " -> " << NidsEngine::H1_DIM
              << " -> " << NidsEngine::NUM_CLASSES << ")" << std::endl;

    std::cout << "[2/4] Pre-loading test dataset into memory: " << test_path << " ... " << std::flush;
    auto samples = load_dataset(test_path, max_samples);
    if (samples.empty()) {
        std::cerr << "No samples loaded." << std::endl;
        return 1;
    }
    std::cout << "[OK] (" << samples.size() << " samples loaded)" << std::endl;

    // Cache warm-up
    std::cout << "[3/4] Warming up CPU caches ... " << std::flush;
    for (size_t i = 0; i < std::min<size_t>(2000, samples.size()); ++i) {
        engine.predict(samples[i].indices.data(), samples[i].indices.size());
    }
    std::cout << "[OK]" << std::endl;

    // Sequential Single-Core Latency Evaluation
    std::cout << "[4/4] Running single-core sequential latency profiling ... " << std::flush;
    std::vector<double> latencies_us;
    latencies_us.reserve(samples.size());
    int correct_predictions = 0;

    for (const auto& sample : samples) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int pred = engine.predict(sample.indices.data(), sample.indices.size());
        auto t_end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        latencies_us.push_back(us);
        if (pred == sample.true_label) {
            correct_predictions++;
        }
    }
    std::cout << "[OK]" << std::endl;

    // Calculate percentiles
    std::sort(latencies_us.begin(), latencies_us.end());
    size_t n = latencies_us.size();
    double median_us = latencies_us[n * 50 / 100];
    double p90_us = latencies_us[n * 90 / 100];
    double p99_us = latencies_us[n * 99 / 100];
    double p999_us = latencies_us[n * 999 / 1000];
    double min_us = latencies_us.front();
    double max_us = latencies_us.back();

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean_us = sum / n;
    double variance = 0.0;
    for (double v : latencies_us) variance += (v - mean_us) * (v - mean_us);
    double stddev_us = std::sqrt(variance / n);

    double accuracy = 100.0 * correct_predictions / n;
    double single_core_throughput = 1000000.0 / mean_us; // windows/sec
    double packets_per_sec = single_core_throughput * 10.0; // 10 packets per window

    std::cout << "\n-------------------- SINGLE-CORE LATENCY RESULTS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Evaluated Samples      : " << n << std::endl;
    std::cout << "  Sample Accuracy        : " << accuracy << "% (" << correct_predictions << "/" << n << ")" << std::endl;
    std::cout << "  Mean Latency           : " << mean_us << " us" << std::endl;
    std::cout << "  Std Dev                : " << stddev_us << " us" << std::endl;
    std::cout << "  Min Latency            : " << min_us << " us" << std::endl;
    std::cout << "  Median (p50) Latency   : " << median_us << " us" << std::endl;
    std::cout << "  90th Percentile (p90)  : " << p90_us << " us" << std::endl;
    std::cout << "  99th Percentile (p99)  : " << p99_us << " us" << std::endl;
    std::cout << "  99.9th Percentile(p99.9): " << p999_us << " us" << std::endl;
    std::cout << "  Throughput (1 Core)    : " << static_cast<uint64_t>(single_core_throughput) << " windows/sec" << std::endl;
    std::cout << "  Equivalent Packet Rate : " << static_cast<uint64_t>(packets_per_sec) << " pkts/sec (10 pkts/window)" << std::endl;

    // Multi-Core Burst Throughput with OpenMP
    int num_threads = omp_get_max_threads();
    std::cout << "\n----------------- MULTI-CORE BURST THROUGHPUT (" << num_threads << " Threads) ----------------" << std::endl;
    auto t_burst_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        engine.predict(samples[i].indices.data(), samples[i].indices.size());
    }
    auto t_burst_end = std::chrono::high_resolution_clock::now();
    double burst_sec = std::chrono::duration<double>(t_burst_end - t_burst_start).count();
    double multi_core_throughput = samples.size() / burst_sec;
    double multi_core_packets = multi_core_throughput * 10.0;

    std::cout << "  Total Processing Time  : " << burst_sec << " s" << std::endl;
    std::cout << "  Multi-Core Throughput  : " << static_cast<uint64_t>(multi_core_throughput) << " windows/sec" << std::endl;
    std::cout << "  Multi-Core Packet Rate : " << static_cast<uint64_t>(multi_core_packets) << " pkts/sec" << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
