# Similar Products Analysis — Intrusion Detection Systems
## COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

---

## Overview

The AT2 feedback recommended moving from *narrative description* to *critical evaluation*: not just
what each system does, but why it is or is not the right fit for the specific context of this project.
This document provides a structured comparison of six IDS solutions across dimensions directly relevant
to the project's research objectives, followed by a critical evaluation of each.

The six systems evaluated are:

| # | System | Type | Developer |
|---|--------|------|-----------|
| 1 | Snort 3 | Signature-based NIDS | Cisco / open-source |
| 2 | Suricata 7 | Signature + rule-based NIDS | OISF / open-source |
| 3 | Zeek (formerly Bro) | Network security monitor | Zeek community / open-source |
| 4 | Darktrace Enterprise | Commercial AI/ML NIDS | Darktrace Ltd |
| 5 | Elastic SIEM (with ML) | Log-based SIEM + ML anomaly | Elastic NV |
| 6 | **This project (AI-IDS)** | Hybrid ML NIDS (RF + IF) | B00979380 / open-source |

---

## Structured Comparison Table

| Dimension | Snort 3 | Suricata 7 | Zeek | Darktrace | Elastic SIEM | **AI-IDS** |
|---|---|---|---|---|---|---|
| **Detection approach** | Signature | Signature + stats | Behavioural scripting | Deep learning (unsupervised) | Rule + ML anomaly | RF (supervised) + IF (anomaly) |
| **ML-based** | ✗ | Partial | ✗ (scripts) | ✓ (proprietary) | Partial | ✓ (full) |
| **Supervised learning** | N/A | N/A | N/A | ✗ | ✗ | ✓ Random Forest |
| **Unsupervised anomaly** | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ Isolation Forest |
| **Explainability (XAI)** | ✗ | ✗ | ✗ | ✗ (black box) | Partial | ✓ SHAP values |
| **Open source** | ✓ (GPLv2) | ✓ (GPLv2) | ✓ (BSD) | ✗ (proprietary) | ✓ (basic tier) | ✓ (MIT) |
| **Requires training data** | ✗ (rules only) | ✗ (rules only) | ✗ (scripts) | ✓ (self-learns) | ✓ (ML module) | ✓ (CICIDS2017) |
| **REST API** | ✗ | ✗ | ✗ | ✓ (enterprise) | ✓ | ✓ `/predict`, `/predict/batch` |
| **Live dashboard** | ✗ (needs external) | ✗ (needs external) | ✗ (needs external) | ✓ | ✓ (Kibana) | ✓ `/dashboard` |
| **Concept drift handling** | Manual rule updates | Manual rule updates | Script updates | Continuous self-learning | Manual retraining | Documented; manual retraining |
| **CICIDS2017 benchmark** | N/A | N/A | N/A | Not published | N/A | ✓ F1-macro 0.567 (RF) |
| **Deployment** | On-premise | On-premise | On-premise | Cloud / on-premise | Cloud / on-premise | Docker / `docker-compose up` |
| **Zero-day detection** | ✗ (no matching rule) | Partial (anomaly rules) | ✗ (script-based) | ✓ (behavioural) | Partial | ✓ (Isolation Forest) |
| **Academic reproducibility** | ✓ | ✓ | ✓ | ✗ (closed source) | ✗ (proprietary ML) | ✓ (full pipeline in Git) |

---

## Critical Evaluation

### 1. Snort 3 (Cisco)

Snort is the most widely deployed open-source NIDS and the de facto baseline for signature-based
detection (Roesch, 1999). Its rule language is highly expressive and its community rulesets
(Talos Intelligence) are updated in near-real-time for known CVEs.

**Why it is not the right fit for this project's context:**
Snort operates entirely on hand-crafted signatures, requiring an analyst to write a rule for every
attack variant. It cannot detect zero-day attacks or novel attack patterns with no prior signature.
Hoover and Mattern (2023) demonstrated that Snort 3 achieves near-perfect recall on known attacks
but 0% recall on zero-day variants in a controlled lab environment — precisely the failure mode this
project's Isolation Forest component addresses. Snort also provides no ML explainability and no
built-in REST API, making it unsuitable for the programmable, self-explaining IDS objective of this
project.

---

### 2. Suricata 7 (OISF)

Suricata extends the Snort rule model with multi-threaded processing, protocol-level deep packet
inspection, and limited statistical anomaly detection through threshold rules. Its JSON output
format makes integration with SIEMs more straightforward than Snort.

**Why it is not the right fit:**
Like Snort, Suricata's core detection remains signature-dependent. Its anomaly detection is limited
to rule-defined thresholds (e.g. "flag as suspicious if more than N packets per second"), not
learned statistical baselines. The project's requirement for a model that adapts to previously
unseen attack patterns — evidenced by the Isolation Forest's ability to detect DoS variants it was
never trained on — cannot be met by Suricata without substantial custom scripting. Additionally,
Suricata offers no explainability layer, which was an explicit project objective (Notebook 07).

---

### 3. Zeek (formerly Bro)

Zeek is a network analysis framework rather than a traditional IDS. It extracts rich protocol-level
metadata (connection logs, DNS logs, HTTP logs, SSL certificates) and exposes them via an
event-driven scripting language. Researchers have used Zeek logs as input features for ML classifiers
(PMC, 2022), but Zeek itself performs no classification.

**Why it is not the right fit:**
Zeek is a data extraction and scripting platform, not a classifier. Integrating ML into Zeek
requires writing custom event handlers in its domain-specific language — a significant engineering
overhead beyond the project's scope. More critically, Zeek does not produce the bidirectional flow
features (packet lengths, inter-arrival times, flag counts) that CICFlowMeter generates for
CICIDS2017. A direct comparison on the same benchmark is therefore not possible. Zeek would be a
strong candidate for a production follow-up system but is out of scope for this academic project.

---

### 4. Darktrace Enterprise

Darktrace uses unsupervised deep learning (specifically, variational autoencoders and recurrent
neural networks) to model a "pattern of life" for each device and user, flagging deviations as
anomalies (Darktrace, 2024). It is the most technically advanced commercial IDS and the closest
competitor to this project's hybrid approach.

**Why it is not the right fit for an academic project:**
Darktrace is a closed, proprietary system. Its models cannot be inspected, reproduced, or benchmarked
against public datasets such as CICIDS2017. The "black box" problem is well-documented: security
analysts report frustration with unexplained alerts, as the system cannot convey *why* a specific
flow was flagged (Frontiers in AI, 2025). This directly motivated the SHAP explainability component
in Notebook 07 — a capability Darktrace lacks at the analyst level. Furthermore, at enterprise
pricing (estimated £50,000–£200,000+ per year), Darktrace is entirely inaccessible for academic
research or small-to-medium enterprise deployment.

**What this project does better:** Full SHAP-based explainability per prediction, open-source
reproducibility, and a documented benchmark against CICIDS2017 — none of which Darktrace provides.

---

### 5. Elastic SIEM (with ML Jobs)

Elastic Security provides log aggregation, correlation rules, and ML anomaly detection jobs
(using isolation forest and rare term analysis) through the Kibana interface. It is widely deployed
in enterprise environments as a SIEM platform.

**Why it is not the right fit:**
Elastic's ML module operates on pre-aggregated log fields, not raw packet-level flow features. It
cannot process the 78 CICFlowMeter features directly, requiring a separate data pipeline to bridge
packet capture to Elasticsearch. Its ML models are trained within the Elastic ecosystem and are not
exportable or inspectable outside of it. The Elastic stack also requires significant infrastructure
(minimum 3-node cluster recommended for production), making it disproportionately complex for a
focused academic IDS project. While its Kibana dashboard is more feature-rich than this project's
built-in dashboard, the underlying ML transparency is no better than Darktrace.

---

### 6. This Project (AI-IDS) — Positioning

This project occupies a specific niche that none of the above systems fills:

| Capability | Where this project leads |
|---|---|
| SHAP explainability on network flows | Only this project provides per-flow feature attribution |
| Hybrid supervised + unsupervised | RF handles known attacks; IF detects zero-days |
| Open-source, reproducible, benchmarked | Full pipeline committed; results reproducible with `docker-compose up` |
| REST API + live dashboard | Only this project ships both out of the box without external tooling |
| Academic benchmark (CICIDS2017) | Published F1-macro results directly comparable to literature |

The primary limitation is dataset age: as discussed in `10_critical_analysis.ipynb`, CICIDS2017 is
seven years old and the trained models would require retraining on current traffic captures before
production deployment. Neither Snort nor Suricata have this limitation because they use human-curated
rules, not learned statistical patterns — which is simultaneously their key advantage and their
fundamental weakness against zero-day attacks.

---

## Summary Comparison Matrix (for report figure)

| System | Detects Zero-Days | Explainable | Open Source | REST API | Dashboard | Benchmark Published |
|---|---|---|---|---|---|---|
| Snort 3 | ✗ | ✗ | ✓ | ✗ | ✗ | N/A |
| Suricata 7 | Partial | ✗ | ✓ | ✗ | ✗ | N/A |
| Zeek | ✗ | ✗ | ✓ | ✗ | ✗ | N/A |
| Darktrace | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Elastic SIEM | Partial | Partial | Partial | ✓ | ✓ | ✗ |
| **AI-IDS (this project)** | **✓** | **✓ (SHAP)** | **✓** | **✓** | **✓** | **✓** |

---

## References

- Roesch, M. (1999) 'Snort — Lightweight intrusion detection for networks', *Proceedings of USENIX LISA*, pp. 229–238.
- Hoover, C. and Mattern, T. (2023) 'Comparative Study of Snort 3 and Suricata Intrusion Detection Systems', *University of Arkansas Computer Science & Computer Engineering Undergraduate Honors Theses*, 105.
- Frontiers in Artificial Intelligence (2025) 'A systematic review on the integration of explainable artificial intelligence in intrusion detection systems to enhancing transparency and interpretability in cybersecurity', *Frontiers in AI*, doi:10.3389/frai.2025.1526221.
- National Center for Biotechnology Information (2022) 'Detecting Reconnaissance and Discovery Tactics from the MITRE ATT&CK Framework in Zeek Conn Logs Using Spark's Machine Learning', *Sensors*, 22(20), 7999.
- Darktrace (2024) *State of AI Cyber Security 2024*. Available at: https://www.darktrace.com/resources/state-of-ai-cyber-security-2024-executive-summary (Accessed: June 2026).
- Stamus Networks (2024) *Suricata vs Snort*. Available at: https://www.stamus-networks.com/suricata-vs-snort (Accessed: June 2026).
