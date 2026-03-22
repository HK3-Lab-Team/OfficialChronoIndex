# ChronoIndex

ChronoIndex is a schema-first toolkit for turning heterogeneous and messy CGM/medical data into coherent, analysis-ready tables.

In real-world studies, preprocessing and unification can become the most burdensome part of the pipeline. This repository was built to validate incoming data rigorously and structure it in a consistent format for downstream analytics.

## Why This Repository Exists

CGM and clinical datasets usually arrive with:

- different identifiers (`PtID`, `id`, `participantId`, `subject_id`, ...)
- different time formats and granularities
- different missing-value patterns and quality issues
- different schema conventions across centers and vendors

ChronoIndex addresses this with a strict but practical workflow.

## Main Blocks

1. Data ingestion and validation (Pydantic-based schemas)
2. Harmonization into shared semantic blocks
3. Unification into cross-dataset tables
4. CGM processing and downstream statistical modules
5. LLM-oriented downstream integration (professional onboarding model)

### 1) Data Ingestion (Pydantic Schemas)

Dataset ingestion is implemented in `chronoindex/dataset_ingestion/` with schema definitions such as:

- `chronoindex/dataset_ingestion/aleppo_data_schemas.py`
- `chronoindex/dataset_ingestion/colas_data_schemas.py`
- `chronoindex/dataset_ingestion/zhao_data_schemas.py`
- `chronoindex/dataset_ingestion/praes_data_schemas.py`

This layer validates and standardizes raw dataset payloads before any cross-dataset operations.

### 2) Harmonization

Dataset-specific harmonizers map validated inputs into common blocks (`CGMData`, `MetaData`, `LabData`, `Food`, `VisitTimepoints`, ...).

- Harmonizers live in `chronoindex/dataset_harmonization/`
- Dataset handlers and harmonizers are configured in `datasets.toml`

### 3) Unification

Unified views are produced in `chronoindex.dataset_unifier` through:

- `UnifiedCGMDataset`
- `UnifiedClinicalData`
- `UnifiedFoodData`
- `UnifiedPhysicalActivityData`

### 4) CGM Processing and Statistical Modules

Downstream utilities are in:

- `chronoindex/glucose_series_processing/`
- `chronoindex/statistical_summaries/`
- `chronoindex/cli/`

This includes quality checks, iglu-style metrics, catch22 features, and plotting.

### 5) LLM-Based Pipeline (Professional Access)

The repository includes a dedicated LLM module (`chronoindex/llm_module`) for professional downstream workflows (for example semantic food annotation and structured generation pipelines).

Because this workflow depends on paid model usage and project-specific integration effort, access is provided through professional onboarding and customer-funded deployment.

## CLI Usage

### Validate Config and Load Datasets

```bash
python -m chronoindex.cli.load_all_datasets --config datasets.toml --dry-run
python -m chronoindex.cli.load_all_datasets --config datasets.toml
```

Load only selected datasets:

```bash
python -m chronoindex.cli.load_all_datasets \
  --config datasets.toml \
  --only aleppo colas zhao
```

### Save Unified Outputs

Unified CGM:

```bash
python -m chronoindex.cli.load_all_datasets \
  --config datasets.toml \
  --save-unified artifacts/unified_cgm.parquet \
  --unified-type cgm
```

Unified clinical:

```bash
python -m chronoindex.cli.load_all_datasets \
  --config datasets.toml \
  --save-unified artifacts/unified_clinical.parquet \
  --unified-type clinical
```

Unified food:

```bash
python -m chronoindex.cli.load_all_datasets \
  --config datasets.toml \
  --save-unified artifacts/unified_food.parquet \
  --unified-type food
```

Unified physical activity:

```bash
python -m chronoindex.cli.load_all_datasets \
  --config datasets.toml \
  --save-unified artifacts/unified_physical_activity.parquet \
  --unified-type physical-activity
```

### Unified CGM Quality Checks

From existing unified parquet:

```bash
python -m chronoindex.cli.cgm_quality_checks \
  --unified-parquet artifacts/unified_cgm.parquet \
  --min-unique-days 14 \
  --min-records 100 \
  --save-report-dir artifacts/cgm_quality_reports
```

From config (build unified CGM internally):

```bash
python -m chronoindex.cli.cgm_quality_checks \
  --config datasets.toml \
  --only aleppo colas zhao
```

### Statistical Description and Metrics (CLI)

You do not need to run the unification step to compute statistical summaries.

If you already have a `.parquet` or `.csv` CGM table, you can pass it directly with `--input-file` as long as it contains:

- `Id` (participant identifier)
- `CGM` (glucose values)
- `CGMTime` (timestamps)

Example with external input (no unification required):

```bash
python -m chronoindex.cli.iglu_metrics \
  --metric summary \
  --input-file data/my_cgm_table.csv \
  --save-output artifacts/summary_stats.parquet
```

More metrics:

```bash
python -m chronoindex.cli.iglu_metrics \
  --metric gmi \
  --input-file data/my_cgm_table.parquet
```

```bash
python -m chronoindex.cli.iglu_metrics \
  --metric in-range \
  --input-file data/my_cgm_table.parquet \
  --target-range 70 180 \
  --target-range 63 140
```

From config (internal unification path):

```bash
python -m chronoindex.cli.iglu_metrics \
  --metric cv \
  --config datasets.toml \
  --only aleppo colas zhao
```

### catch22 Features

```bash
python -m chronoindex.cli.catch22_features \
  --input-file data/my_cgm_table.parquet \
  --flatten \
  --save-output artifacts/catch22_features.parquet
```

Subject-level aggregation without timestamp grouping:

```bash
python -m chronoindex.cli.catch22_features \
  --input-file data/my_cgm_table.parquet \
  --no-time-col \
  --save-output artifacts/catch22_subject_level.parquet
```

### Plotting

Daily subject time series:

```bash
python -m chronoindex.cli.plot \
  --plot time-series \
  --level subject \
  --input-file data/my_cgm_table.parquet \
  --subject-id s24417854 \
  --daily \
  --save-dir artifacts/plots
```

Dataset-level frequency distribution:

```bash
python -m chronoindex.cli.plot \
  --plot frequency \
  --level dataset \
  --input-file data/my_cgm_table.parquet \
  --save-dir artifacts/plots
```

## Python API Quick Start

```python
from chronoindex.dataset_loader import load_datasets
from chronoindex.dataset_unifier import UnifiedCGMDataset

datasets, errors = load_datasets(config_path="datasets.toml")
if errors:
    print(errors)

ucgm = UnifiedCGMDataset(
    datasets,
    metadata_columns=["Age", "Type of Diabetes", "BMI", "HbA1c"],
)
ucgm.unified_data.write_parquet("artifacts/unified_cgm.parquet")
```

## Extending ChronoIndex with a New Dataset

1. Define dataset schemas (`*_data_schemas.py`)
2. Implement a dataset adapter from `base_cgm_dataset.py`
3. Add harmonization logic in `chronoindex/dataset_harmonization/`
4. Register handler and harmonizer in `datasets.toml`
5. Run `--dry-run`, then full load, then unified export

## License

This repository is distributed under the terms described in the [LICENSE](LICENSE) file.

This codebase is part of [**HK3Lab**](https://hk3lab.ai/)’s contribution to the [PRAESIIDIUM project](https://cordis.europa.eu/project/id/101095672), funded by the European Union’s Horizon Europe research and innovation program under grant agreement No 101095672.

**Disclaimer**: The content reflects the work of the authors only. The European Commission is not liable for any use that may be made of the information or tools contained herein.

<p align="center">
  <img src="Images/header-logo.png" alt="MedSchemaGen Header" width="100"/>
</p>
