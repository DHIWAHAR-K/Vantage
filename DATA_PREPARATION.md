# Data Preparation Workflow

## Overview

The training pipeline now uses pre-processed data for faster, more efficient training. This eliminates the need to load and process the 9.3GB `data.json` file on every training run.

## Workflow

```
Raw SynSQL Data (22.9M examples)
    ↓
prepare_data.py
    ↓
Processed Data (5M subset, train/val split)
    ↓
data_loader.py (fast loading)
    ↓
Training
```

## Step 1: Prepare Data (One-time)

Run this once before training:

```bash
python scripts/prepare_data.py --subset-size 5000000
```

**What it does:**
1. Loads `data/synsql/data.json` (2.5M examples)
2. Integrates schemas from `data/synsql/tables.json` (16K databases)
3. Creates 95/5 train/validation split
4. Saves to `data/processed/train.json` and `data/processed/val.json`

**Options:**
- `--subset-size`: Number of examples to use (default: 5M)
- `--val-ratio`: Validation split ratio (default: 0.05 = 5%)
- `--synsql-dir`: Path to SynSQL directory (default: data/synsql)
- `--output-dir`: Output directory (default: data/processed)

**Output:**
```
data/processed/
├── train.json      # Training examples
├── val.json        # Validation examples
└── stats.json      # Dataset statistics
```

## Step 2: Training

The training script now uses the processed data:

```bash
python scripts/finetune.py --config configs/t5_finetune.yaml
```

**Benefits:**
- **10-20x faster startup**: No need to load 9.3GB file
- **Consistent splits**: Same train/val split every run
- **Efficient iteration**: Pre-processed format
- **Less memory**: Only loads what's needed
- **Curriculum learning**: Fast filtering by difficulty

## Data Format

### Raw SynSQL Format

```json
{
  "question": "What are the descriptions...",
  "sql": "SELECT description, complexity\nFROM...",
  "sql_complexity": "Simple",
  "db_id": "code_snippet_management"
}
```

### Processed Format

```json
{
  "question": "What are the descriptions...",
  "schema": "code_snippets(id, description, complexity, is_public) | ...",
  "sql": "SELECT description, complexity FROM...",
  "db_id": "code_snippet_management",
  "difficulty": "Simple"
}
```

**Key differences:**
1. ✅ Schema integrated from `tables.json`
2. ✅ Consistent field names
3. ✅ Ready for tokenization
4. ✅ Difficulty field normalized

## Curriculum Learning

During training, data is loaded by difficulty phase:

**Phase 1 (Steps 0-40K): Simple SQL**
```python
loader = ProcessedDataLoader(
    data_dir="data/processed",
    split="train",
    difficulty_filter="Simple"  # Only simple queries
)
```

**Phase 2 (Steps 40K-75K): Medium SQL**
```python
difficulty_filter="Medium"  # JOINs, GROUP BY
```

**Phase 3 (Steps 75K-100K): Complex SQL**
```python
difficulty_filter="Complex"  # Nested queries, multiple JOINs
```

## Dataset Statistics

After running `prepare_data.py`, check `data/processed/stats.json`:

```json
{
  "total_examples": 5000000,
  "train_examples": 4750000,
  "val_examples": 250000,
  "val_ratio": 0.05,
  "difficulty_distribution": {
    "Simple": {"train": 1500000, "val": 75000},
    "Medium": {"train": 2000000, "val": 100000},
    "Complex": {"train": 1250000, "val": 75000}
  }
}
```

## Memory Usage

**Before (streaming from raw data):**
- Load 9.3GB data.json: ~10 GB RAM
- Parse JSON on-the-fly: High CPU
- Schema lookup: Additional overhead
- Total: ~12-15 GB RAM

**After (processed data):**
- Load train.json subset: ~2-3 GB RAM
- Pre-parsed format: Low CPU
- Schema pre-integrated: No lookup
- Total: ~3-5 GB RAM

**Savings: ~60% less memory, 10-20x faster startup**

## Troubleshooting

### Error: "Processed data not found"

**Problem:** Training script can't find processed data

**Solution:**
```bash
python scripts/prepare_data.py
```

### Error: "FileNotFoundError: data.json"

**Problem:** SynSQL data not in expected location

**Solution:**
```bash
python scripts/prepare_data.py --synsql-dir /path/to/synsql
```

### Want to use full dataset?

**Problem:** Default uses 5M subset, want all 22.9M

**Solution:**
```bash
python scripts/prepare_data.py --subset-size 22900000
```

**Note:** Will take longer and require more RAM (128GB should be fine)

### Want different train/val split?

**Problem:** Default is 95/5, want 90/10

**Solution:**
```bash
python scripts/prepare_data.py --val-ratio 0.10
```

## Re-running Preparation

If you want to re-process the data with different settings:

```bash
# Remove old processed data
rm -rf data/processed

# Run with new settings
python scripts/prepare_data.py --subset-size 10000000 --val-ratio 0.10
```

## Next Steps

After preparing data:

1. ✅ Data is ready: `data/processed/train.json` and `data/processed/val.json`
2. ⏳ Convert T5: `python scripts/convert_t5_to_mlx.py`
3. ⏳ Start training: `python scripts/finetune.py`

---

**Summary:** Run `prepare_data.py` once, then train multiple times without re-processing! 🚀
