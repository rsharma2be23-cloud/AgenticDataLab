# src/agents/eda_agent.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools

# optional import for VIF
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

sns.set(style="darkgrid")


class EDAAgent:
    """
    EDAAgent: compute descriptive stats, correlation matrices (pearson/spearman/kendall),
    compute VIF when available, compute skew/kurtosis, generate histograms/heatmaps,
    store outputs via FileTools and MemoryTools with a stable schema.
    """

    def __init__(self, output_dir="eda_outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.file_tool = FileTools()
        self.memory = MemoryTools()

    def _safe_write_file(self, path, content_bytes):
        """
        Write bytes via FileTools if write_binary exists, else fallback to text write marker.
        """
        try:
            if hasattr(self.file_tool, "write_binary"):
                return self.file_tool.write_binary(path, content_bytes)
            else:
                # fallback: write a small marker with the path
                try:
                    return self.file_tool.write(path, f"FILE_SAVED:{path}")
                except Exception:
                    return {"status": "error", "error": "filetool_write_failed"}
        except Exception:
            return {"status": "error", "error": "filetool_write_exception"}

    def _save_fig(self, fig, filename):
        """ Save matplotlib figure locally and attempt to store via FileTools. """
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        # attempt to push binary
        try:
            with open(path, "rb") as f:
                data = f.read()
            self._safe_write_file(path, data)
        except Exception:
            # ignore write failures, but path still returned
            pass
        return path

    def _compute_vif(self, df_numeric):
        """Compute VIF per numeric feature if statsmodels is available."""
        if not _HAS_STATSMODELS or df_numeric.shape[1] < 2:
            return {"status": "skipped", "reason": "statsmodels_not_available_or_too_few_columns"}

        try:
            X = df_numeric.dropna()
            # add constant for VIF calculation if needed
            Xc = sm.add_constant(X)
            vifs = {}
            for i, col in enumerate(X.columns):
                # variance_inflation_factor expects ndarray; we skip constant at index 0
                vif_val = variance_inflation_factor(Xc.values, i + 1)  # +1 because const at 0
                vifs[col] = float(np.round(vif_val, 4))
            return {"status": "success", "vif": vifs}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run(self, df: pd.DataFrame):
        """
        Run EDA and return a structured result containing:
          - describe (all columns)
          - numeric_columns
          - categorical_columns
          - correlations: {pearson, spearman, kendall} (each a dict)
          - corr_image_paths (for each type if generated)
          - corr_interactive_path (optional static image)
          - missing_path
          - histograms: list of file paths
          - boxplots: list of file paths
          - outlier_counts (IQR)
          - skewness/kurtosis per numeric column
          - vif (if available)
        Saves outputs to memory under key "eda_output".
        """
        try:
            # basic describe
            describe_stats = df.describe(include="all").fillna("").to_dict()

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            # histograms & boxplots
            hist_paths = []
            box_paths = []
            for col in numeric_cols:
                # histogram
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histogram: {col}")
                hist_path = self._save_fig(fig, f"hist_{col}.png")
                hist_paths.append(hist_path)

                # boxplot
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=df[col].dropna(), ax=ax2)
                ax2.set_title(f"Boxplot: {col}")
                box_path = self._save_fig(fig2, f"box_{col}.png")
                box_paths.append(box_path)

            # missing heatmap
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(df.isnull(), cmap="viridis", cbar=False, ax=ax)
            missing_path = self._save_fig(fig, "missing_heatmap.png")

            # correlation matrices (pearson, spearman, kendall)
            correlations = {}
            corr_image_paths = {}
            if len(numeric_cols) >= 2:
                numeric_df = df[numeric_cols]
                for typ in ("pearson", "spearman", "kendall"):
                    try:
                        if typ == "pearson":
                            corr = numeric_df.corr(method="pearson").round(4).fillna(0)
                        elif typ == "spearman":
                            corr = numeric_df.corr(method="spearman").round(4).fillna(0)
                        else:
                            corr = numeric_df.corr(method="kendall").round(4).fillna(0)

                        correlations[typ] = corr.to_dict()

                        # save heatmap image
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                        ax.set_title(f"{typ.title()} Correlation")
                        img_path = self._save_fig(fig, f"corr_{typ}.png")
                        corr_image_paths[typ] = img_path
                    except Exception as e:
                        correlations[typ] = {"error": str(e)}
            else:
                correlations = {"status": "not_enough_numeric_cols"}

            # compute outliers via IQR
            outlier_counts = {}
            for col in numeric_cols:
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
                except Exception:
                    outlier_counts[col] = None

            # skewness & kurtosis
            skew_kurt = {}
            for col in numeric_cols:
                try:
                    skew_kurt[col] = {
                        "skewness": float(np.round(df[col].dropna().skew(), 4)),
                        "kurtosis": float(np.round(df[col].dropna().kurtosis(), 4))
                    }
                except Exception:
                    skew_kurt[col] = {"skewness": None, "kurtosis": None}

            # VIF
            vif_res = self._compute_vif(df[numeric_cols]) if len(numeric_cols) >= 2 else {"status": "skipped", "reason": "too_few_cols"}

            result = {
                "status": "success",
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "describe": describe_stats,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "histograms": hist_paths,
                "boxplots": box_paths,
                "missing_path": missing_path,
                "correlations": correlations,
                "corr_image_paths": corr_image_paths,
                "outlier_counts": outlier_counts,
                "skew_kurtosis": skew_kurt,
                "vif": vif_res,
            }

            # save JSON summary file and to memory
            summary_path = os.path.join(self.output_dir, "eda_summary.json")
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                # attempt to store via file tool
                try:
                    with open(summary_path, "rb") as f:
                        self._safe_write_file(summary_path, f.read())
                except Exception:
                    pass
            except Exception:
                pass

            # store to memory under stable key
            try:
                self.memory.save("eda_output", result)
            except Exception:
                # Memory save failure shouldn't break pipeline
                pass

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}
