# M2-Isala-Spatio-Temporal-Dispersion-Ablation-Categorization-

Python utilities for classifying patients treated with Spatio Temporal Dispersion Ablation,
including supervised/unsupervised grouping and recurrence quantification.

## Install

```bash
pip install -r requirements.txt
```

## Example

```python
from stda_classification import (
    load_patient_dataset,
    classify_patients_supervised,
    cluster_patients_unsupervised,
    quantify_recurrence_by_group,
)

# CSV or MATLAB (.mat) datasets exported from CARTO/PentaRay workflows.
df = load_patient_dataset("patient_signals.csv")
features = ["fractionation_index", "cycle_length_ms", "signal_entropy"]

metrics = classify_patients_supervised(df, features, target_column="recurrence")
clustered = cluster_patients_unsupervised(df, features, n_clusters=3)
recurrence_summary = quantify_recurrence_by_group(clustered)
```
