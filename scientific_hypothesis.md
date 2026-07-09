# Project Hypothesis: SLIDE Viability in NIDS

This document registers our primary scientific hypothesis regarding the computational viability and classification performance of the SLIDE (LSH-Sparse Neural Network) algorithm in comparison to standard dense machine learning models for Network Intrusion Detection Systems (NIDS).

---

## 1. Primary Scientific Hypothesis

### Statement:
> For low-dimensional NIDS datasets (under 10,000 features), standard dense CPU-optimized models (like PyTorch MLPs or XGBoost) will outperform SLIDE in both training speed and accuracy. However, as the input feature space scales past **10,000+ sparse dimensions** (driven by raw packet headers and temporal sliding windows), SLIDE will become the only deep learning algorithm capable of performing line-rate packet classification on standard CPU hardware without packet loss.

### The Crossover Point:
*   **Below Crossover (Low Dimensions):** Standard matrix multiplication libraries (using AVX-512 vectorization) easily cache the entire network weights. The mathematical overhead of LSH makes SLIDE slower.
*   **Above Crossover (High Dimensions / High Sparsity):** The dense weight matrix grows too large for CPU cache limits. SLIDE's LSH layer scales sub-linearly, bypassing 99% of the computations and maintaining line-rate throughput.

---

## 2. Optimal Window Size: Sparsity, Security, and Accuracy

We do not need to use an extremely long time window (such as $N=100$ packets) which would complicate testing and delay classification. Instead, we aim for a **compromise window of $N=5$ to $N=15$ packets** (yielding $6,535$ to $19,605$ input dimensions).

### Why this is the optimal compromise:
1.  **Immediate Signal Density (Security):** The vast majority of network attacks reveal their signatures during the connection handshake and the initial request headers (e.g., HTTP request, TLS Client Hello, DNS request/response). This initial payload phase happens entirely within the **first 5 to 10 packets** of a flow.
2.  **Low Latency (Security):** A window of 10 packets is processed in milliseconds. Buffering 10 packet headers for classification introduces no noticeable lag to the end-user, whereas buffering 100 packets would cause significant delays.
3.  **High Sparsity (Performance):** For $N=10$, we have $13,070$ dimensions with exactly **60 active features** (99.54% sparse). This is sparse enough to benefit from SLIDE's LSH while keeping the model's memory footprint extremely small ($\approx 53$ MB), allowing it to fit entirely within the CPU's L2/L3 cache.
4.  **Testing Simplicity (Accuracy):** Extracting and testing the first 10 packets of a flow from existing PCAP datasets is straightforward. It avoids the complexity of simulating long-term sequence dynamics or writing complex zero-padding testing engines for 100-packet prefixes.

---

## 3. The Accuracy vs. Latency/Throughput Trade-off

A critical question is whether a model using only **6 features per packet over a 10-packet window** (Version 4) can compete with standard NIDS models that achieve **97-98% accuracy** using 47 complex statistical features.

### Why Standard Models Achieve 98% Accuracy (The Post-Facto Caveat):
Standard NIDS models rely on highly aggregated, flow-based statistical features (like `smean` (mean packet size), `sjit` (jitter), `dur` (duration), or `sload` (bits per second)). These features are extremely descriptive, but **they can only be calculated after the connection has closed or after buffering millions of packets.** 
*   **The Security Failure:** By the time the model has gathered enough packets to calculate "mean packet size" and flags the connection as an exploit, the payload has already executed and the system is breached.

### Why the Sparse Time-Window Model is Viable:
Although each packet in Version 4 only has 6 features, the **sequence of these features** contains the protocol's state machine:
*   Instead of looking at the static average packet size, the model inspects the exact order of events: `[SYN -> SYN-ACK -> ACK -> HTTP GET -> Payload -> FIN]`.
*   Anomalies (like a SYN flood, port scan, or SQL injection) break this sequence immediately (e.g., multiple consecutive SYNs, unexpected payload size on a specific port, or anomalous protocol transitions).
*   **The Verdict:** While a sequence-based model might achieve a slightly lower raw accuracy (e.g. 88%–93%) compared to offline statistical models, **it runs in real-time and detects attacks instantly (e.g., at packet 5), allowing the firewall to actively block the threat before it completes.** 

In network security, an 88% accuracy model that blocks attacks in real-time at 10 Gbps is vastly superior to a 98% accuracy model that runs at only 100 Mbps and detects attacks too late.
