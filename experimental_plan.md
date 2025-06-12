
# 🧪 Experimental Plan: AI-Driven Data Quality Monitoring

This experimental plan is designed to assess both **performance** and **validity** of the AI-powered data quality monitoring system across different configurations and scenarios.

---

## ✅ Performance Variables
These variables affect latency, cost, and overall system performance.

### 1. Model Variants
Evaluate LLM behavior across model types:

- `GPT-4o`
- `GPT-4o-mini`
- `GPT-3.5`
- `GPT-4.1`

### 2. Prompting & Validation Features
Assess prompt engineering techniques and validation tools:

- **RCI Prompting**: Rule/note generation enriched with reasoning and self-checks
- **Final Validation**: Post-run reasoning pass to validate rule/note coherence
- **Note Validation**: Functionality check of LLM-generated notes
- **Domain Extraction**: Auto-extract column-level metadata for context-aware prompting

### 3. Dataset Size
Impact of dataset scale on performance:

- `Small` — representative sample
- `Medium` — realistic production size
- `Large` — stress test with scale

#### Goals:
- Assess impact on:
  - Violation detection accuracy
  - System latency
  - Token and cost metrics

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

### Automation Tips:
- Use a script to log metrics to a `.csv` or `.jsonl` per run
- Log each config as a unique `run_id` with all relevant metadata
- Capture versioning for prompt templates and datasets

---

## 📂 Suggested Folder Structure

```
/experiments/
  ├── configs/
  │   ├── gpt4o_small_health.json
  │   └── gpt35_large_sensor.json
  ├── results/
  │   ├── run_001.json
  │   └── run_002.json
  ├── logs/
  │   └── runtime_log.csv
  └── prompts/
      ├── base_prompt_v1.txt
      └── rci_prompt_v2.txt
```

---

## 📌 Next Steps

- [ ] Convert this plan into a Linear board with tasks per cycle
- [ ] Finalize logging schema for metrics collection
- [ ] Automate run tracking (shell + Python logger)
- [ ] Begin first benchmark cycle with `GPT-4o` on `Retail` dataset


---

## 🔁 Repeat Runs for Drift & Stability Analysis

To measure **drift** and ensure consistency of LLM outputs, each experiment configuration should be executed **5 times**.

### 🔹 Purpose
- Capture stochastic variation in LLM outputs
- Validate system **stability and reproducibility**
- Identify flaky behavior in rule/note generation or anomaly detection

### 🧮 Metrics to Aggregate
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
| `drift_score`  | (Optional) Avg. difference between output masks  |

---

### 🗃️ Example Result JSON Schema

```json
{
  "config_id": "gpt4o_rci_health_small_0.01",
  "model": "GPT-4o",
  "prompting": "RCI",
  "dataset": "Health",
  "size": "Small",
  "injection_type": "Missing",
  "injection_level": 0.01,
  "runs": [
    { "run_id": 1, "f1": 0.82, "precision": 0.85, "recall": 0.79, "tokens": 5800, "latency_sec": 12.4 },
    { "run_id": 2, "f1": 0.80, "precision": 0.84, "recall": 0.76, "tokens": 5700, "latency_sec": 12.1 },
    ...
  ],
  "aggregates": {
    "mean_f1": 0.81,
    "std_f1": 0.015,
    "mean_latency": 12.6,
    "mean_tokens": 5700
  }
}
```

---

### 🧰 Automation Tip
- Use Python or shell scripts to repeat and log runs automatically
- Store logs in a structured format (`.jsonl` or `.csv`) for easy aggregation
