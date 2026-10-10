"""Small, UI-independent helpers for validating uploaded datasets."""
from io import BytesIO
import csv
import io

import pandas as pd


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DATASET_ROWS = 100_000


def read_uploaded_dataset(upload, max_bytes=MAX_UPLOAD_BYTES, max_rows=MAX_DATASET_ROWS):
    """Read a bounded CSV upload. The uploader is never written to a user path."""
    name = getattr(upload, "name", "")
    if not isinstance(name, str) or not name.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file with a .csv extension.")
    content = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds the {max_bytes // (1024 * 1024)} MiB upload limit.")
    try:
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))), None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("CSV must use valid UTF-8 text encoding.") from exc
    if not header or any(not str(name).strip() for name in header):
        raise ValueError("The CSV must contain non-empty column names.")
    if len(set(header)) != len(header):
        raise ValueError("Column names must be unique.")
    try:
        frame = pd.read_csv(BytesIO(content), nrows=max_rows + 1)
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must use valid UTF-8 text encoding.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        # Parser exceptions can echo raw CSV cells; avoid placing dataset content in UI errors.
        raise ValueError("Could not parse this CSV. Check that it is a well-formed comma-separated file.") from exc
    if len(frame) > max_rows:
        raise ValueError(f"Dataset exceeds the {max_rows:,} row limit.")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The CSV must contain a header and at least one data row.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame
