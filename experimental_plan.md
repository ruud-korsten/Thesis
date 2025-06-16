
# 🧪 Experimental Plan: AI-Driven Data Quality Monitoring

This experimental plan is designed to assess both **performance** and **validity** of the AI-powered data quality monitoring system across different configurations and scenarios.

---

## ✅ Performance Variables
These variables affect latency, cost, and overall system performance.

### 1. Model Variants
Evaluate LLM behavior across model types:

- `GPT-4o-mini`
- `GPT-4.1`
- `deepseek-chat`
- `deepseek-reasoner`
- `Mistral`



### 2. Prompting & Validation Features
Assess prompt engineering techniques and validation tools:

- **RCI Prompting**: Rule/note generation enriched with reasoning and self-checks
- **Final Validation**: Post-run reasoning pass to validate rule/note coherence
- **Note Validation**: Functionality check of LLM-generated notes
- **Domain Extraction**: Auto-extract column-level metadata for context-aware prompting


### 3. Dataset Size
Impact of dataset scale on performance:

- `Small` — 1,000 rows
- `Medium` — 50,000 rows
- `Large` — 500,000 rows

These tiers reflect increasingly realistic production environments. Large-scale tests (≥100k rows) allow for stress-testing the LLM pipeline under volume. While 1 million rows could be considered for extreme-scale benchmarking, 100k strikes a balance between runtime feasibility and realism for most mid-sized data operations.

#### Goals:
- Assess impact on:
  - Violation detection accuracy
  - System latency and runtime
  - Token usage and cost metrics


---

## 🧪 Validity Variables
These variables assess detection reliability across different domains and error scenarios.

### 1. Datasets (Domain Coverage)
Evaluate detection robustness across real-world verticals:

- `Retail`
- `Health`
- `Wallmart`
- `Sensor`

### 2. DQ Injection Types
Simulated error types and levels to test detection sensitivity:

#### Error Types:
- `Missing values`
- `Type mismatches`
- `Duplicates`
- `Outliers`

#### Injection Levels:
- `0.0` — Clean baseline
- `0.01` — Low corruption
- `0.1` — High corruption

Each combination tests both false positives and true positive recall of anomalies.

---

## 📏 Metrics
Each experiment run collects metrics across four dimensions:

### 1. Speed
- Total pipeline runtime
- Per-stage latency (loading, LLM calls, validation, etc.)

### 2. Token & Cost
- Tokens used per model stage
- Approximate cost per run based on OpenAI pricing

### 3. Accuracy
- Compare predicted vs. ground truth **violation masks**
- Binary classification metrics (Precision, Recall, F1)

### 4. Drift
- Inconsistencies across equivalent runs (same config, different seeds)
- Captures model randomness or instability

---

## 🔁 Execution Structure
Recommended approach for running experiments:

### Iteration Cycles:
1. **Fix Performance Variables**, vary Validity Variables
2. **Fix DQ Scenario**, vary Prompting/Model settings
3. **Repeat for all model variants and dataset sizes**


### 4. Control Baseline
To quantify the added value of Generative AI, a non-LLM baseline is included for comparison.

#### Baseline Methods:
- **Static Rules Only**: Apply predefined standard checks (e.g., missing values, type mismatches, duplicates) without LLM involvement.
- **Open-source DQ Tools**: Use tools like `great_expectations` to define and validate expectations on the same datasets.

These baselines act as reference points for evaluating the **relative performance of LLM-enhanced pipelines**, helping to isolate the impact of generative components on anomaly detection precision, recall, and cost.

#### Evaluation Goals:
- Measure how much LLMs improve rule coverage and accuracy.
- Assess whether the extra compute cost is justified by measurable gains.
- Provide interpretability comparisons between traditional and AI-generated outputs.


---

## 🔁 Repeat Runs for Drift & Stability Analysis

To measure **drift** and ensure consistency of LLM outputs, each experiment configuration is executed **5 times**.

### 🔹 Sliding Window Strategy
Instead of sampling randomly with fixed seeds, we apply a **sliding window technique** on the dataset to increase realism and variance sensitivity:

- For each dataset tier (e.g., small = 1000 rows), run N uses the first N × 1000 rows:
  - **Run 1:** rows 1–1000
  - **Run 2:** rows 1–2000
  - **Run 3:** rows 1–3000
  - ...
  - **Run 5:** rows 1–5000

This method:
- Simulates progressive system load
- Exposes temporal or pattern-based biases
- Helps observe scaling behavior even within one tier

Note: when the dataset has fewer rows than the max window size, padding or looping can be applied, or runs may be capped accordingly.

### Metrics to Aggregate
For each configuration (model + prompt + dataset + error type + injection level):

1. **Run 5 times**
2. Log per-run:
   - `F1`, `Precision`, `Recall`
   - `Token usage`
   - `Runtime`

3. Compute:

| Metric         | Description                                      |
|----------------|--------------------------------------------------|
| `mean_f1`      | Average F1 score across 5 runs                   |
| `std_f1`       | Standard deviation of F1 (drift indicator)       |
| `mean_latency` | Average runtime                                  |
| `mean_tokens`  | Average token usage                              |
| `drift_score`  | Avg. difference between output masks             |

To measure **drift** and ensure consistency of LLM outputs, each experiment configuration should be executed **5 times**.

### Purpose
- Capture stochastic variation in LLM outputs
- Validate system **stability and reproducibility**
- Identify flaky behavior in rule/note generation or anomaly detection

### Metrics to Aggregate
For each configuration (model + prompt + dataset + error type + injection level):

1. **Run 5 times**
2. Log per-run:
   - `F1`, `Precision`, `Recall`
   - `Token usage`
   - `Runtime`

3. Compute:

| Metric         | Description                                      |
|----------------|--------------------------------------------------|
| `mean_f1`      | Average F1 score across 5 runs                   |
| `std_f1`       | Standard deviation of F1 (drift indicator)       |
| `mean_latency` | Average runtime                                  |
| `mean_tokens`  | Average token usage                              |
| `drift_score`  | Avg. difference between output masks             |
