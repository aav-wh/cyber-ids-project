# AT3 — Product Demo Script
## COM668 | Abdulbosit Abdurazzakov | B00979380
## 15 minutes | 25% of grade
## Framing: Product pitch + live demo — examiners must understand WHAT, WHY, and WHY BUY IT

---

## THE GOLDEN THREAD (say this to yourself before recording)

> "I identified a real problem in cybersecurity. I built a working AI product that solves it.
> The product is live, it detects attacks in real time, and I can prove it works."

Every sentence in this video serves that thread.

---

## TIMED STRUCTURE

| # | Section | Time | Cumulative |
|---|---------|------|------------|
| 1 | The Problem — why this matters | 1:30 | 1:30 |
| 2 | The Solution — what I built | 1:00 | 2:30 |
| 3 | LIVE DEMO — Dashboard walkthrough | 8:00 | 10:30 |
| 4 | Why it works — key decisions justified | 2:00 | 12:30 |
| 5 | Real-world relevance — market context | 0:45 | 13:15 |
| 6 | Honest limitations + what's next | 1:00 | 14:15 |
| 7 | Closing pitch | 0:45 | 15:00 |

---

## FULL SCRIPT

---

### [0:00 – 1:30] THE PROBLEM

*[Face camera. Speak with confidence. No slides needed — just talk.]*

> "Every 39 seconds, a cyberattack happens somewhere in the world.
> In 2023 alone, cybercrime cost the global economy over 8 trillion dollars.
> And the average time to detect a breach? 204 days.
>
> The reason detection takes so long is simple — traditional rule-based intrusion detection systems can't keep up. Attackers evolve faster than the rules can be written. They need something smarter.
>
> My project asks: can we use machine learning to detect network intrusions automatically, in real time, without needing a human to write every rule?
>
> The answer is yes — and I built it."

---

### [1:30 – 2:30] THE SOLUTION

*[Still facing camera or show a clean title screen with your name.]*

> "What I built is an AI-Based Intrusion Detection System — a complete, end-to-end product.
>
> It was trained on CICIDS2017 — a real network traffic dataset captured over 8 days, containing 2.8 million network flows and 15 different attack types including DDoS, port scanning, brute-force, and web attacks.
>
> The system uses not one but two AI models — Random Forest for supervised classification and Isolation Forest for unsupervised anomaly detection — combined in an ensemble. They work together like two independent analysts checking each other's work.
>
> The result is a live, running product — a professional threat intelligence dashboard — that can classify any network flow as an attack or benign traffic in under 10 milliseconds.
>
> Let me show you."

*[Open browser. Dashboard should be on Threat Visualizer page.]*

---

### [2:30 – 10:30] LIVE DEMO — 8 MINUTES

---

#### [2:30 – 4:30] THE DASHBOARD — Threat Visualizer

*[Full screen on the browser. Walk slowly.]*

> "This is the Threat Visualizer — the command centre of the system.
>
> The first thing you notice is the threat level banner at the top — showing ELEVATED, with a live indicator. This gives a security analyst an immediate situational awareness the moment they open the product. No hunting through logs.
>
> These four cards give the key numbers at a glance: 9 completed pipeline stages, 97% peak detection rate, and the challenge I had to solve — the 85% miss rate I'll explain shortly.
>
> On the left — this is what makes this product look and feel like a real security tool. A live network topology map. Every circle is a device on the network. Red glowing nodes are compromised hosts — the system has flagged them as high threat. Amber is suspicious. Cyan is normal. You can instantly see which part of your network is under attack.
>
> On the right — a live detection feed. Every row is a classified network flow with a timestamp, host, description, and threat score. This is what a security operations analyst would monitor in real time.
>
> Below — a 24-hour attack frequency chart. You can see attack volume peaks at certain hours — this kind of temporal pattern is exactly what a human analyst would miss but an AI system catches automatically."

---

#### [4:30 – 5:30] MODEL PERFORMANCE PAGE

*[Click "Model Intelligence"]*

> "This page shows the evidence that the system works.
>
> But here's what's interesting — the headline accuracy number doesn't tell the full story.
> My Random Forest model achieves 99% precision. Sounds perfect, right?
>
> But look at the recall — 14.9%. That means at the default settings, the model misses 85% of all attacks. It's like a security guard who never gives false alarms, but sleeps through 8 out of every 10 real break-ins.
>
> This is the central challenge I had to solve — and I solved it."

*[Click the Threshold tab]*

> "This PR curve shows how I fixed it. By tuning the decision threshold on the Isolation Forest model — moving it from the default to minus 0.125 — the recall jumps to 97%. The system now catches 97 out of every 100 attacks.
>
> This is the kind of decision that separates an academic exercise from a real product. Real IDS systems prioritise not missing attacks over avoiding false alarms — because a missed attack is a breach, and a false alarm is just an investigation."

---

#### [5:30 – 7:00] LIVE THREAT DETECTION — Threat Hunt

*[Click "Threat Hunt"]*

> "Now let me show the product in action — live classification.
>
> This page lets you submit any network flow and get an instant verdict from all three detection engines simultaneously.
>
> I'm going to simulate a DDoS attack — 500,000 packets per second, no return traffic, targeting port 80."

*[Select DDoS flood. Click Classify Flow.]*

> "Classified in milliseconds. All three engines agree — ATTACK.
> The gauge shows 90%+ attack probability from the Random Forest model.
>
> Now let me show normal traffic — a standard HTTP web browsing session."

*[Select Benign HTTP. Click Classify Flow.]*

> "Instantly classified as BENIGN. Green across the board. The system correctly distinguishes between legitimate traffic and an active attack.
>
> In a real deployment, this would be running on every single flow on your network — thousands per second — automatically."

---

#### [7:00 – 8:30] BATCH ANALYSIS

*[Click "Batch Analysis"]*

> "In the real world, security teams don't investigate one flow at a time. They need to analyse traffic in bulk — forensic analysis after an incident, or screening captured traffic from a suspicious time window.
>
> This is the Batch Analysis engine. Upload any CSV of network flows, and the system classifies every single row."

*[Upload sample_flows.csv]*

> "I've prepared 70 network flows — a realistic mix of normal web traffic, DDoS patterns, port scans, and SSH brute-force attempts.
>
> The system processes all 70 in seconds. The results show exactly how many threats were detected, the breakdown between attack and benign traffic, and a full results table you can download for further analysis.
>
> This is the kind of feature a real enterprise security team would use after receiving a threat intelligence report — bulk-screen their captured traffic to see if the attack pattern appeared in their network."

---

#### [8:30 – 10:00] LIVE DETECTION SIMULATION

*[Click "Live Detection"]*

> "This page simulates what the product looks like in a live deployment scenario.
>
> I'm generating 60 synthetic network flows with a 25% attack ratio — simulating a network under active attack."

*[Set sliders. Click Run Detection Stream.]*

> "Watch the stream process in real time — each flow being classified as it arrives, just like a live network tap.
>
> The results are clear: the system caught the majority of attacks, with recall and precision shown directly. The scatter plot shows every flow plotted by its attack probability — red attacks cluster at the top, blue benign flows at the bottom, with a clear separation.
>
> The flows that were missed — shown at the bottom — directly validate the threshold tuning work. With the PR-optimal threshold applied, these misses reduce by over 80%."

---

#### [10:00 – 10:30] SHAP — Why did it flag that?

*[Click "AI Explainability"]*

> "One critical question for any AI security product: why did the system flag that flow?
>
> A black-box detector is useless in court, useless in an incident report, and dangerous if it's wrong. My system includes SHAP explainability — every decision can be traced back to specific features.
>
> The top feature? Destination Port. Certain ports — 22 for SSH, 3389 for Remote Desktop, 445 for SMB file sharing — are known attack vectors. The model learned this from the data, not from a human writing rules.
>
> This is what makes the system trustworthy."

---

### [10:30 – 12:30] WHY IT WORKS — KEY DECISIONS

*[Face camera or stay on dashboard.]*

> "Let me explain three design decisions that make this product work.
>
> **First — two models, not one.**
> Random Forest is a supervised model — excellent at recognising attack patterns it's seen before. Isolation Forest is unsupervised — it detects anomalies even in traffic it's never seen. Together in an ensemble using an OR rule — flag if either detects — they catch threats that either model alone would miss. This mirrors how real security systems work: multiple independent sensors.
>
> **Second — threshold tuning.**
> Most ML tutorials stop at training the model and reporting accuracy. I went further. I used precision-recall curve analysis to find the optimal decision threshold — moving from 85% miss rate to 3% miss rate. This single change transformed the product from academically interesting to operationally useful.
>
> **Third — preventing data leakage.**
> This is a subtle but critical engineering decision. When I applied SMOTE — a technique to balance the imbalanced dataset — I applied it only to the training data, after the train/test split. If you apply it before the split, test data leaks into training, and your metrics look great but the model is cheating. My metrics are real."

---

### [12:30 – 13:15] REAL-WORLD RELEVANCE

> "To put this in context — the commercial product this most closely resembles is Darktrace, which uses unsupervised machine learning for network anomaly detection. Darktrace IPO'd on the London Stock Exchange valued at 1.7 billion pounds.
>
> Vectra AI, which uses a similar supervised-plus-unsupervised approach to my ensemble, raised over 200 million dollars in funding.
>
> My project independently arrived at the same architectural decisions — dual-model ensemble, anomaly detection, explainability — that billion-dollar companies built their products on. That's validation that the approach is sound."

---

### [13:15 – 14:15] HONEST LIMITATIONS

> "I want to be honest about three limitations.
>
> First — CICIDS2017 is from 2017. The dataset doesn't include modern attack types: AI-generated phishing, supply-chain attacks, or cloud-native threats. A production system would need retraining on current traffic.
>
> Second — concept drift. I tested this specifically. A model trained on Monday's traffic performs worse on Friday's traffic. Attack patterns shift over time, and a real deployment needs continuous retraining, not a static model.
>
> Third — the system works on extracted flow features. In a real network, you'd need a flow exporter like CICFlowMeter sitting on a network tap to generate the 78 features the model expects. That's the integration gap between research and production.
>
> These aren't failures — they're the known boundary conditions of the system, and identifying them honestly is part of good engineering."

---

### [14:15 – 15:00] CLOSING PITCH

> "What I've shown you today is a working AI product — not a proof of concept, not a Jupyter notebook, but a running system.
>
> It detects network attacks in under 10 milliseconds. It explains every decision. It handles single flows, batch analysis, and live simulation. It has 52 automated tests ensuring it doesn't break. And it's built on the same architectural principles as commercially successful security products.
>
> The problem is real. The solution works. And it's running right now.
>
> Thank you."

---

## RECORDING TIPS

- **Record section by section** — don't try to do 15 minutes in one take
- **Speak slower than feels natural** — on camera everything sounds faster
- **Open the dashboard full screen** — hide bookmarks bar (Ctrl+Shift+B)
- **Don't apologise** — if you stumble, pause and continue. No "sorry, let me redo that"
- **Name the file**: `B00979380_COM668_AT3.mp4`
- **The 3 questions examiners ask themselves:**
  1. Do I understand what this does? ✓ (product demo)
  2. Do I believe it works? ✓ (live classification)
  3. Does the student understand why they made these choices? ✓ (design decisions section)
