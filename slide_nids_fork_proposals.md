# Long-Term Proposals: NIDS-Specialized SLIDE Fork (SLIDE-NIDS)

This document registers the long-term, high-level proposals for specializing and optimizing the SLIDE (Locality Sensitive Hashing based Neural Network) algorithm for Network Intrusion Detection Systems (NIDS). 

Each proposal is detailed from a high-level architectural perspective and includes a priority/impact rating evaluating its relevance for **performance** (throughput, latency, CPU utilization) and **accuracy** (precision, recall, false alarm rate).

---

## 1. Streaming C++ Preprocessing Pipeline

### Description:
Network security devices must process packets and connection logs in real-time at the line rate. The original SLIDE implementation was designed for batch, offline training and assumes the input is already preprocessed into text-based formats (like CSV or SVMLight) and loaded entirely into memory. This proposal integrates a lightweight, high-performance packet/log parser directly into SLIDE's C++ entry point.

### Core Design Elements:
*   **Socket Listener:** The C++ binary opens a Unix/TCP socket to listen directly to streaming logs (e.g., Zeek/Bro JSON output, NetFlow streams, or OS audit logs).
*   **In-Memory Feature Map:** A fast C++ preprocessing module parses incoming records, applies logarithmic binning boundaries for numeric features, and maps strings (like protocols and services) directly to active index coordinates in a pre-allocated sparse vector.
*   **Zero-Copy Loading:** The preprocessed sparse vectors are fed directly to the LSH hashing layer in-memory, bypassing any disk writing/reading operations.

### Priority & Impact Rating:
*   **Performance Impact:** **CRITICAL / HIGH**
    *   *Why:* Eliminates disk I/O bottlenecks. Writing telemetry to disk and reading it back as SVMLight files is the largest bottleneck in NIDS pipelines. Real-time in-memory parsing enables SLIDE to evaluate traffic with sub-millisecond latency.
*   **Accuracy Impact:** **LOW**
    *   *Why:* It preserves the exact same numeric representation and category mapping; it simply accelerates how they reach the model.
*   **Overall Priority:** **High** (Essential for real-world deployment).

---

## 2. Cache-Optimized LSH for Temporal Locality (Connection Caching)

### Description:
Network traffic exhibits extreme temporal locality: packets belonging to the same TCP connection or UDP session arrive consecutively or within short windows. The original SLIDE algorithm computes LSH hash values and performs hash table lookups for every single input vector. This proposal introduces an in-memory cache that stores the computed active neurons of recently seen active flows.

### Core Design Elements:
*   **Flow Tracker Cache:** Maintain a small, highly optimized LRU (Least Recently Used) cache keyed by flow identifiers (e.g., `src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto`).
*   **Bypass Mechanism:** When a packet/flow update arrives, the engine checks the cache:
    *   *Cache Hit:* The engine retrieves the pre-calculated active hidden layer neuron indices associated with that active connection. It skips LSH hash computations and table lookups entirely, directly proceeding to weight updates/forward pass.
    *   *Cache Miss:* The engine computes LSH hashes, performs the lookups, updates the hidden neuron set, and caches the result for future packets.

### Priority & Impact Rating:
*   **Performance Impact:** **HIGH**
    *   *Why:* LSH hash computations (multiple mathematical hash functions) and hash table lookups are CPU-intensive. By caching active connections, we can bypass LSH lookups for 80-90% of packets in long-lived flows (like video streams, file downloads, or SSH sessions), freeing up massive CPU capacity.
*   **Accuracy Impact:** **NEGLIGIBLE**
    *   *Why:* A connection's baseline OHE features (like port and protocol) remain constant for its duration. Caching preserves the mathematical correctness of these dimensions.
*   **Overall Priority:** **High** (Crucial for high-throughput packet processing).

---

## 3. Adaptive Inference Sparsity (CPU Safety Valve)

### Description:
Under normal conditions, a NIDS can afford to run deep, complex evaluations. However, during network anomalies or flooding attacks (like SYN floods, UDP floods, or DDoS), packet rates spike exponentially. This proposal enables SLIDE to dynamically adjust its sparsity parameter ($k$, the number of active hidden neurons computed) in response to system load, acting as an automated CPU safety valve.

### Core Design Elements:
*   **System Load Monitor:** Monitor the size of the packet intake queue or CPU utilization on a separate lightweight thread.
*   **Dynamic $k$ Scaling:**
    *   *Low Traffic:* Set $k$ high (e.g., evaluate top-128 neurons) for maximum precision and accuracy.
    *   *High Traffic (Congestion):* Automatically scale $k$ down (e.g., evaluate top-16 neurons) to accelerate feedforward calculations, sacrificing minor diagnostic detail to maintain line rate.
    *   *DDoS State:* Drop LSH table updates entirely and use basic heuristics or ultra-sparse forward passes to flag flooding sources without crashing the host CPU.

### Priority & Impact Rating:
*   **Performance Impact:** **CRITICAL / HIGH**
    *   *Why:* A NIDS must remain resilient under attack. If the NIDS crashes or lags under a DDoS, the network either goes down or is left completely blind. This feature guarantees survivability of the defense system.
*   **Accuracy Impact:** **MEDIUM** (Temporary drop during congestion)
    *   *Why:* Reducing the number of evaluated active neurons slightly degrades the model's capacity to distinguish subtle, complex attacks, but keeps the system active to detect obvious flood sources.
*   **Overall Priority:** **Medium-High** (Critical for defense robustness).

---

## 4. Built-in Cost-Sensitive Loss & Asymmetric Decision Boundaries

### Description:
Standard neural networks treat all errors equally during training (e.g., misclassifying `Worms` as `Normal` carries the same penalty as misclassifying `Worms` as `DoS`). In security, failing to flag an attack (False Negative) is catastrophic, whereas flagging a normal connection as an attack (False Positive) is a manageable alert. This proposal integrates cost-sensitive learning natively into SLIDE's C++ backpropagation and activation thresholding.

### Core Design Elements:
*   **Cost-Weighted Gradient:** Modify the loss gradient calculation inside the C++ engine to scale the backpropagation error by a penalty weight loaded from a Cost Matrix.
*   **Defensive Thresholding Layer:** Implement a custom classification layer at the output of the network. Instead of a simple `argmax()` over softmax probabilities, the model applies a lower threshold for attack categories. If any attack probability exceeds, say, 15%, the system defaults to flagging it as an attack, prioritizing caution.

### Priority & Impact Rating:
*   **Performance Impact:** **NEGLIGIBLE**
    *   *Why:* Multiplying the output delta by a constant weight during backpropagation adds a single floating-point operation per batch, causing no measurable delay.
*   **Accuracy Impact:** **HIGH**
    *   *Why:* Directly shifts the model's precision-recall curve to heavily minimize False Negatives, making the system highly reliable as an intrusion firewall.
*   **Overall Priority:** **High** (Aligns the ML algorithm's mathematics with security-industry requirements).

---

## 5. Concept Drift Mitigation via Online Continuous Learning

### Description:
Network traffic is dynamic: new devices are added, software updates change packet signatures, and user behaviors shift. An offline-trained classifier will experience "concept drift" and decay in accuracy over time. This proposal implements a background online learning loop in C++ to adapt the model to shifting baselines without requiring complete offline retraining.

### Core Design Elements:
*   **Low-Priority Training Thread:** Dedicate a low-priority background thread to run continuous online backpropagation updates.
*   **High-Confidence Weight Updates:** When the inference engine classifies a connection as `Normal` with extreme confidence (e.g., $> 99.9\%$), it passes that sample to the background training queue to update the model's weights on the active network profile.
*   **Signature Lock:** Freeze the weights associated with specific critical attack signatures (e.g., `Exploits`, `DoS`) to prevent the model from "forgetting" how to detect malicious behaviors due to catastrophic forgetting.

### Priority & Impact Rating:
*   **Performance Impact:** **MEDIUM**
    *   *Why:* Background threads consume CPU cycles, which must be carefully throttle-controlled so they do not compete with the primary packet classification thread.
*   **Accuracy Impact:** **HIGH**
    *   *Why:* Maintains high detection rates and keeps false alarms low over months of deployment, solving the major real-world limitation of static machine learning classifiers.
*   **Overall Priority:** **Medium** (Crucial for long-term production feasibility, but can be implemented in later stages).
