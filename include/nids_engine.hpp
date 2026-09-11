#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <Eigen/Dense>

/**
 * @brief High-performance, production-ready NIDS inference engine using Eigen 3.
 * 
 * Features:
 * - Direct execution of sparse categorical input vectors via column-gather on W0.
 * - Dense GEMV for hidden layers optimized with native SIMD (AVX2/FMA3).
 * - Zero dynamic heap allocations during inference.
 * - Flat binary weight loader (zero external library dependencies).
 */
class NidsEngine {
public:
    static constexpr int INPUT_DIM = 2910;
    static constexpr int H0_DIM = 2048;
    static constexpr int H1_DIM = 256;
    static constexpr int NUM_CLASSES = 15;

    // Weight matrices
    // W0 is ColMajor: Column j is contiguous in memory for AVX2 vector addition.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> W0;
    Eigen::VectorXf b0;

    // W1 and W2 are RowMajor for cache-friendly GEMV.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W1;
    Eigen::VectorXf b1;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W2;
    Eigen::VectorXf b2;

    bool is_loaded = false;

    NidsEngine() = default;

    /**
     * @brief Loads model weights from flat binary file (.bin).
     */
    bool load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[NidsEngine] Error: Failed to open weight file: " << filepath << std::endl;
            return false;
        }

        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "NIDS") {
            std::cerr << "[NidsEngine] Error: Invalid file magic." << std::endl;
            return false;
        }

        uint32_t header[5];
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        uint32_t version = header[0];
        uint32_t in_dim = header[1];
        uint32_t h0_dim = header[2];
        uint32_t h1_dim = header[3];
        uint32_t num_cls = header[4];

        if (in_dim != INPUT_DIM || h0_dim != H0_DIM || h1_dim != H1_DIM || num_cls != NUM_CLASSES) {
            std::cerr << "[NidsEngine] Error: Dimension mismatch in binary file." << std::endl;
            return false;
        }

        W0.resize(h0_dim, in_dim);
        b0.resize(h0_dim);
        W1.resize(h1_dim, h0_dim);
        b1.resize(h1_dim);
        W2.resize(num_cls, h1_dim);
        b2.resize(num_cls);

        // Read raw contiguous arrays directly into Eigen memory buffers
        file.read(reinterpret_cast<char*>(W0.data()), W0.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b0.data()), b0.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));

        if (!file) {
            std::cerr << "[NidsEngine] Error: Incomplete read of weight tensors." << std::endl;
            return false;
        }

        is_loaded = true;
        return true;
    }

    /**
     * @brief Performs deterministic single-window inference.
     * 
     * @param active_indices Array of active feature indices in [0, INPUT_DIM).
     * @param num_active Number of active indices.
     * @param out_logits Optional pointer to receive the 15 output logits.
     * @return Predicted class ID (argmax).
     */
    inline int predict(const int* active_indices, int num_active, float* out_logits = nullptr) const {
        // Layer 0: Column gather (Input Sparsity) + Bias (8 KB on stack)
        Eigen::Matrix<float, H0_DIM, 1> h0 = b0;
        for (int i = 0; i < num_active; ++i) {
            int idx = active_indices[i];
            if (idx >= 0 && idx < INPUT_DIM) {
                h0.noalias() += W0.col(idx);
            }
        }
        // Activation: ReLU
        h0 = h0.cwiseMax(0.0f);

        // Layer 1: Dense GEMV + ReLU (1 KB on stack)
        Eigen::Matrix<float, H1_DIM, 1> h1 = (W1 * h0 + b1).cwiseMax(0.0f);

        // Layer 2: Output logits (60 B on stack)
        Eigen::Matrix<float, NUM_CLASSES, 1> logits = W2 * h1 + b2;

        if (out_logits) {
            for (int c = 0; c < NUM_CLASSES; ++c) {
                out_logits[c] = logits[c];
            }
        }

        int max_idx = 0;
        logits.maxCoeff(&max_idx);
        return max_idx;
    }
};
