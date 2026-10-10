"""Dataset diagnostics that report only checks computed from the supplied data."""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataQualityResult:
    quality_score: int
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self):
        return {"status": "success", **asdict(self)}


class DataQualityAgent:
    """Inspect a dataframe before modeling; score is a transparent issue penalty."""
    def analyze(self, df, target_col=None):
        issues, warnings, recommendations = [], [], []
        if not isinstance(df, pd.DataFrame) or df.empty:
            return DataQualityResult(0, ["Dataset is empty or is not a DataFrame."], [], [], {}).to_dict()

        duplicate_rows = int(df.duplicated().sum())
        duplicate_columns = int(df.columns.duplicated().sum())
        missing_counts = df.isna().sum()
        missing = {str(c): int(v) for c, v in missing_counts.items() if v}
        numeric = [col for pos, col in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df.iloc[:, pos])]
        categorical = [col for pos, col in enumerate(df.columns)
                       if (pd.api.types.is_object_dtype(df.iloc[:, pos]) or
                           pd.api.types.is_string_dtype(df.iloc[:, pos]) or
                           isinstance(df.iloc[:, pos].dtype, pd.CategoricalDtype) or
                           pd.api.types.is_bool_dtype(df.iloc[:, pos]))]
        constant = [str(col) for pos, col in enumerate(df.columns)
                    if df.iloc[:, pos].nunique(dropna=False) <= 1]
        near_constant = []
        high_cardinality = []
        suspicious_ids = []
        outliers = {}
        invalid_values = {}
        for position, col in enumerate(df.columns):
            series = df.iloc[:, position]
            n_unique = int(series.nunique(dropna=True))
            ratio = n_unique / max(len(df), 1)
            frequency = series.value_counts(dropna=False, normalize=True)
            if len(frequency) and frequency.iloc[0] >= 0.98 and n_unique > 1:
                near_constant.append(str(col))
            if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or
                    isinstance(series.dtype, pd.CategoricalDtype) or
                    pd.api.types.is_bool_dtype(series)) and n_unique >= 20 and ratio >= 0.5:
                high_cardinality.append(str(col))
            name = str(col).strip().lower()
            if (name in {"id", "key", "uuid", "index"} or name.endswith("_id") or name.startswith("id_")) and ratio >= 0.8:
                suspicious_ids.append(str(col))
            if pd.api.types.is_numeric_dtype(series):
                values = series.dropna().to_numpy()
                count_invalid = int((~np.isfinite(values)).sum()) if len(values) else 0
                if count_invalid:
                    invalid_values[str(col)] = count_invalid
                finite = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite):
                    q1, q3 = finite.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    n_outliers = int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum())
                    if n_outliers:
                        outliers[str(col)] = n_outliers

        if duplicate_rows:
            issues.append(f"{duplicate_rows} duplicate rows detected.")
            recommendations.append("Check whether duplicate records should be removed before training.")
        if duplicate_columns:
            issues.append(f"{duplicate_columns} duplicate column names detected.")
        if missing:
            issues.append(f"Missing values detected in {len(missing)} columns.")
            if any(count / len(df) >= 0.2 for count in missing.values()):
                recommendations.append("Review imputation or missingness patterns before training.")
        if constant:
            issues.append(f"{len(constant)} constant columns detected.")
            recommendations.append("Remove constant columns before training.")
        if near_constant:
            warnings.append(f"Near-constant columns: {', '.join(near_constant)}.")
        if invalid_values:
            issues.append("Infinite or non-finite numeric values detected.")
            recommendations.append("Replace or remove non-finite numeric values before training.")
        if high_cardinality:
            warnings.append(f"High-cardinality categorical columns: {', '.join(high_cardinality)}.")
            recommendations.append("Review high-cardinality categories for safe encoding or exclusion.")
        if suspicious_ids:
            warnings.append(f"Potential identifier columns: {', '.join(suspicious_ids)}.")
            recommendations.append("Review suspicious identifier columns and exclude them if they do not generalize.")
        if outliers:
            warnings.append("IQR outliers detected in: " + ", ".join(outliers))

        imbalance = None
        leakage = []
        target_task = None
        target_is_unique = target_col is not None and int((df.columns == target_col).sum()) == 1
        if target_is_unique:
            target = df.loc[:, target_col]
            target_unique = int(target.nunique(dropna=True))
            classification_threshold = min(20, max(2, int(np.sqrt(max(int(target.notna().sum()), 1)))))
            target_task = "classification" if (not pd.api.types.is_numeric_dtype(target) or target_unique <= classification_threshold) else "regression"
            proportions = target.value_counts(normalize=True, dropna=True)
            if target_task == "classification" and len(proportions) > 1:
                imbalance = {"majority_fraction": float(proportions.iloc[0]),
                             "minority_fraction": float(proportions.iloc[-1]),
                             "class_count": int(len(proportions)),
                             "is_imbalanced": bool(proportions.iloc[0] >= 0.8)}
                if proportions.iloc[0] >= 0.8:
                    warnings.append("Strong target class imbalance detected.")
                    recommendations.append("Use class-aware evaluation and consider class_weight=balanced.")
            for pos, col in enumerate(df.columns):
                if col == target_col:
                    continue
                try:
                    series = df.iloc[:, pos]
                    if series.equals(target):
                        leakage.append(str(col))
                    elif pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                        correlation = pd.concat([series.rename("feature"), target.rename("target")], axis=1).corr().iloc[0, 1]
                        if pd.notna(correlation) and abs(float(correlation)) >= 0.995:
                            leakage.append(str(col))
                except (TypeError, ValueError):
                    pass
            if leakage:
                warnings.append("Possible target leakage from: " + ", ".join(leakage))
                recommendations.append("Inspect possible leakage columns before trusting validation scores.")

        # Transparent penalty: issues cost 10, warnings cost 3, bounded to [0,100].
        score = max(0, 100 - 10 * len(issues) - 3 * len(warnings))
        return DataQualityResult(
            quality_score=score, issues=issues, warnings=warnings, recommendations=recommendations,
            checks={"rows": int(len(df)), "columns": int(df.shape[1]), "missing_by_column": missing,
                    "target_task": target_task,
                    "duplicate_rows": duplicate_rows, "duplicate_columns": duplicate_columns,
                    "constant_columns": constant, "near_constant_columns": near_constant,
                    "numeric_columns": [str(c) for c in numeric],
                    "categorical_columns": [str(c) for c in categorical],
                    "high_cardinality_categorical": high_cardinality, "suspicious_id_columns": suspicious_ids,
                    "invalid_numeric_values": invalid_values, "outlier_counts": outliers,
                    "class_imbalance": imbalance, "possible_target_leakage": leakage}
        ).to_dict()
