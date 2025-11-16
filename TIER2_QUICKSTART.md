# Tier 2: Quick Start Guide

Get started with Model Improvement in 5 minutes.

## 1. Feature Engineering (5 min)

Boost baseline RF performance with domain-specific features:

```bash
python3 src/train_best_call_baseline.py \
  --data data/processed/call_states_synth_large.csv \
  --feature-engineering \
  --importance
```

**Output**:
- `models/feature_importance.json` — Top features ranked
- Console: Feature importance table
- `models/cv_report.csv` — CV metrics per fold

**Expected Gain**: +1-3% F1 on synthetic data, +5-15% on real data

---

## 2. Generate Larger Dataset (2 min)

Create 5000+ sample dataset with realistic feature correlations:

```bash
python3 src/generate_large_dataset.py --n 5000 --seed 42
```

**Output**: `data/processed/call_states_large.csv` (5000 rows)

**Why**: More data → better generalization (often +20-30% F1)

---

## 3. Train on Larger Dataset (5-10 min)

```bash
python3 src/train_best_call_baseline.py \
  --data data/processed/call_states_large.csv \
  --feature-engineering --cv
```

**Expected Gain**: +2-4x F1 improvement vs 900-sample baseline

---

## 4. Hyperparameter Tune (30 min - optional)

Automatically find optimal hyperparameters:

```bash
python3 src/tune_multimodal.py \
  --data data/processed/call_states_synth_large.csv \
  --images data/dummy_screenshots \
  --n-trials 20 --cv-folds 3 --epochs 10
```

**Output**:
- `models/tuning_results_YYYYMMDD_HHMMSS.json`
- Console: Recommended params for next training

**Use**: Then train multimodal with best params:
```bash
python3 src/multimodal_model.py \
  --data data/processed/call_states_synth_large.csv \
  --images data/dummy_screenshots \
  --lr [tuned_lr] \
  --batch-size [tuned_bs] \
  --epochs 50
```

---

## 5. Benchmark All Approaches (2-30 min)

**Quick comparison** (2 min):
```bash
python3 src/benchmark_tier2.py --quick
```

**Full comparison** (30 min):
```bash
python3 src/benchmark_tier2.py --full
```

**Output**: 
- Console table with all models
- `models/benchmark_results_*.json`

---

## One-Liner Workflows

### Minimal (5 min)
```bash
python3 src/train_best_call_baseline.py \
  --data data/processed/call_states_synth_large.csv \
  --feature-engineering
```

### Standard (15 min)
```bash
# Generate larger dataset
python3 src/generate_large_dataset.py --n 5000

# Train with FE
python3 src/train_best_call_baseline.py \
  --data data/processed/call_states_large.csv \
  --feature-engineering --importance
```

### Complete (45 min)
```bash
# Generate dataset
python3 src/generate_large_dataset.py --n 5000

# Tune hyperparams
python3 src/tune_multimodal.py \
  --data data/processed/call_states_synth_large.csv \
  --images data/dummy_screenshots \
  --n-trials 20 --epochs 10

# Benchmark
python3 src/benchmark_tier2.py --quick
```

---

## Expected Results

### On 900 synthetic samples:
- RF (no FE): F1 = 0.139
- RF (+FE): F1 = 0.142 (+2%)
- Multimodal: F1 = 0.190 (+37%)

### On 2000+ samples:
- RF (+FE): F1 = 0.549 (+3.9x)

### On real data (estimates):
- FE alone: +5-15%
- Larger dataset: +20-30%
- Combined: 2-3x improvement possible

---

## Troubleshooting

**Memory issues?**
```bash
# Reduce batch size
python3 src/tune_multimodal.py --cv-folds 2 --epochs 5
```

**Want faster results?**
```bash
# Skip multimodal, focus on baseline
python3 src/benchmark_tier2.py --quick
```

**Need specific hyperparams?**
```bash
# Use Optuna results in training
python3 src/tune_multimodal.py --n-trials 30
# Extract best_params from JSON
# Use in multimodal_model.py --lr, --batch-size, etc.
```

---

## Next Steps

1. **Run Phase 1** (Feature Engineering) on your real labeled data
2. **Collect results** — compare F1 before/after
3. **Scale to larger dataset** if you have more data
4. **Fine-tune hyperparameters** for multimodal model
5. **Deploy best model** via REST API

---

## Files to Check After Running

```
models/
├── cv_report.csv              # CV metrics per fold
├── feature_importance.json    # Feature rankings
├── tuning_results_*.json      # Best hyperparams
└── benchmark_results_*.json   # All model comparisons
```

---

## Documentation

- Full docs: `README.md` → Tier 2 section
- Implementation details: `TIER2_SUMMARY.md`
- Project status: `PROJECT_STATUS.md`

---

**Ready? Start with:** `python3 src/train_best_call_baseline.py --data data/processed/call_states_synth_large.csv --feature-engineering`
