import pandas as pd
import numpy as np

class DatasetTools:
    """
    Enhanced utility toolset for Dataset Explorer.
    This version includes:
        - Column profiling
        - Duplicate analysis
        - Auto-dtype detection
    """

    def get_shape(self, df):
        """Return shape of dataframe"""
        try:
            return {"rows": df.shape[0], "cols": df.shape[1]}
        except Exception as e:
            return {"error": str(e)}

    def get_columns(self, df):
        """Return list of column names"""
        try:
            return {"columns": df.columns.tolist()}
        except Exception as e:
            return {"error": str(e)}
        
    # ----------------------------------------------------
    # BASIC SUMMARY
    # ----------------------------------------------------
    def summary(self, df: pd.DataFrame):
        try:
            return {
                "status": "success",
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------
    # COLUMN PROFILE (IMPORTANT)
    # ----------------------------------------------------
    def profile_column(self, df: pd.DataFrame, col: str):
        try:
            series = df[col]
            missing = series.isnull().sum()
            missing_pct = round(missing / len(series) * 100, 2)
            unique = series.nunique()
            sample_vals = series.dropna().unique()[:5].tolist()

            profile = {
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": missing_pct,
                "unique": unique,
                "sample_values": sample_vals
            }

            # Numeric-specific stats
            if pd.api.types.is_numeric_dtype(series):
                profile.update({
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std())
                })

            # Categorical stats
            elif series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
                top_counts = series.value_counts().head(5).to_dict()
                profile.update({"top_values": top_counts})

            return {"status": "success", "profile": profile}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------
    # DUPLICATE ANALYSIS (IMPORTANT)
    # ----------------------------------------------------
    def duplicate_analysis(self, df: pd.DataFrame):
        try:
            total_dupes = df.duplicated().sum()
            duplicated_rows = df[df.duplicated(keep=False)].head(20).to_dict(orient="records")

            per_column = {
                col: df[col].duplicated().sum()
                for col in df.columns
            }

            return {
                "status": "success",
                "total_duplicates": int(total_dupes),
                "sample_duplicate_rows": duplicated_rows,
                "duplicates_per_column": per_column
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------
    # AUTO-DTYPE DETECTION (IMPORTANT)
    # ----------------------------------------------------
    def auto_detect_dtype(self, series: pd.Series):
        s = series.dropna()

        # try datetime
        try:
            pd.to_datetime(s, errors="raise")
            return "datetime"
        except Exception:
            pass

        # try integer
        if s.apply(lambda x: str(x).isdigit()).all():
            return "int"

        # try float
        try:
            s.astype(float)
            return "float"
        except Exception:
            pass

        # try boolean
        bool_set = {"true", "false", "yes", "no", "0", "1"}
        if s.apply(lambda x: str(x).lower() in bool_set).all():
            return "bool"

        # fallback
        return "category"

    def auto_dtype_suggestions(self, df: pd.DataFrame):
        try:
            suggestions = {}
            for col in df.columns:
                if df[col].dtype == "object":
                    suggestions[col] = self.auto_detect_dtype(df[col])

            return {"status": "success", "suggestions": suggestions}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------
    # BASIC FUNCTIONS
    # ----------------------------------------------------
    def missing_values(self, df: pd.DataFrame):
        try:
            return {
                "status": "success",
                "missing": df.isnull().sum().to_dict()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def describe(self, df: pd.DataFrame):
        try:
            desc = df.describe(include="all").fillna("NA").to_dict()
            return {"status": "success", "describe": desc}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def head(self, df: pd.DataFrame, n=5):
        try:
            return {
                "status": "success",
                "head": df.head(n).to_dict(orient="records")
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def correlations(self, df: pd.DataFrame):
        try:
            numeric_df = df.select_dtypes(include="number")
            corr = numeric_df.corr().round(3).fillna(0)
            return {
                "status": "success",
                "correlation_matrix": corr.to_dict()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
