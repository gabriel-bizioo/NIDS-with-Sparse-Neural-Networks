# Project Roadmap: High-Dimensional Sparsification and SLIDE NIDS

This document registers our project objectives, sparsification strategies, and the design for **Version 4** and **Version 5** raw packet header representations.

---

## 1. Project Objectives & Strategies

### Core Objective:
Develop a Network Intrusion Detection System (NIDS) utilizing the **SLIDE** (Locality Sensitive Hashing) algorithm on CPU hardware that performs real-time, packet-level classification at line rate, without the high latency of traditional flow-based feature extraction.

### Core Strategies:
1.  **High Sparsity Representation:** Map packet attributes to high-dimensional, ultra-sparse One-Hot Encoded (OHE) vectors to play into SLIDE's strengths (computing active hidden neurons via LSH).
2.  **Short-Term Baselines:** Test the 3 current versions of the sparsified UNSW-NB15 dataset (v1: 178 dims, v2: 241 dims, v3: 541 dims) to verify execution speed and classification accuracy.
3.  **Advanced Evaluation:** Evaluate models using **Macro-F1 Score** to guarantee that rare security threats (like Worms and Shellcode) are actively detected instead of being ignored by majority-class guessing.
4.  **Security Defense Prioritization:** Implement a **Hierarchical Gate** structure (Binary Gate for Benign/Attack followed by Multiclass Diagnostic) to minimize false negatives (attacks leaking through).

---

## 2. Are IP Addresses Useful in NIDS Inference?

### The Overfitting & Generalization Problem:
If we one-hot encode raw IP addresses (e.g. `192.168.1.15`), the neural network will simply memorize which local IPs are attacks and which are benign. 
*   **Result:** The model will achieve near 100% accuracy on the test set but **fail completely** if deployed on a different network topology where those same IP addresses belong to different devices.
*   **Concept Drift:** Public IPv4 addresses used by attackers change constantly (dynamic botnets, VPNs, IP spoofing). Memorizing specific external IPs leads to immediate accuracy decay.

### The Solution: Role-Based IP Mapping
Instead of raw IP strings, we map IPs to **categorical role contexts** before encoding:
*   **Directionality:** `is_source_internal` and `is_destination_internal` (Binary: 0 or 1).
*   **Local Network Subnet Roles:** Group internal IPs by their functional subnet (e.g. `Server_Subnet`, `IoT_Subnet`, `User_Wi-Fi`, `Gateway`).
*   **Geographical/ASN Grouping:** For external IPs, resolve them to their country or Autonomous System Number (ASN) category.

---

## 3. Version 4 Design: Raw Header & Time-Window Representation

Version 4 shifts from pre-processed CSV datasets to raw network packet headers extracted from live interfaces or PCAP streams. 

### Feature Layout (Per Packet):
1.  **Protocol Type:** `ip_proto` (OHE of IP protocols) $\rightarrow$ **256 dimensions**
2.  **Header Flags:** `tcp_flags` (OHE of SYN, ACK, FIN, RST, PSH, URG, ECE, CWR) $\rightarrow$ **8 dimensions**
3.  **Source Port Type:** `sport_type` (Coarse OHE: System, Registered, Dynamic) $\rightarrow$ **3 dimensions**
4.  **Destination Port:** `dport` (OHE of well-known ports [0-1023] + one dynamic category) $\rightarrow$ **1,025 dimensions**
5.  **IP Subnet Roles:** `src_role` and `dst_role` (Categorical: Local, External) $\rightarrow$ **2 dimensions each (total 4)**
6.  **Packet Length:** `pkt_len` (Binned packet size: Tiny, Small, Medium, Large, Jumbo) $\rightarrow$ **5 dimensions**

*Total dimensions per packet = 1,301 dimensions.*
*Total dimensions over a 10-packet window ($N=10$) = 13,010 dimensions.*
*Active features per sample = 60 (99.54% sparse).*

---

## 4. Version 5 Design: Advanced Raw Header & Flow Dynamics

Version 5 builds upon Version 4 by adding OS fingerprinting indicators and flow dynamics (sequencing).

### Additional Features:
1.  **IP Time to Live (TTL):** `ip_ttl` (OHE of standard TTL brackets: $\le 64$, $65\text{-}128$, $129\text{-}255$, $>255$) $\rightarrow$ **4 dimensions**
    *   *Significance:* Crucial for OS fingerprinting (Linux defaults to 64, Windows to 128) and identifying IP spoofing.
2.  **TCP Sequence Number Delta:** `seq_delta` (Binned difference between current and last packet's sequence number modulo $2^{32}$) $\rightarrow$ **4 dimensions**
    *   *Significance:* Represents the payload size sent in this direction; helps identify TCP sequence prediction attacks.
3.  **TCP Acknowledgment Number Delta:** `ack_delta` (Binned difference between current and last packet's ack number modulo $2^{32}$) $\rightarrow$ **4 dimensions**
    *   *Significance:* Indicates bytes acknowledged; helps track sliding window synchronization and request-response states.

*Total dimensions per packet = 1,313 dimensions.*

### Sliding Window Sequence ($N=10$):
*   **Total Input Dimensions:** $10 \times 1,313 = \mathbf{13,130}$ dimensions.
*   **Active Features:** Exactly **90** active features per sample (9 per packet $\times$ 10 packets).
*   **Sparsity:** **99.31%** sparse.
*   **Generated Script:** [sparsify_pcap_v5.py](file:///home/bizio/Repos/NIDS-with-Sparse-Neural-Networks/scripts/sparsify_pcap_v5.py)
