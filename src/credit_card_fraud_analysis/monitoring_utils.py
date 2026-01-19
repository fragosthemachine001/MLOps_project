import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime, UTC
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset, DatasetStats
from credit_card_fraud_analysis.utils.my_logger import logger
LOG_FILE = Path("prediction_database.csv")

def log_to_database(features: List[float], error: float, is_fraud: bool):
    now = datetime.now(tz=UTC).isoformat()
    data_row = [now] + features + [error, is_fraud]
    df = pd.DataFrame([data_row])

    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", index=False, header=header)

def generate_drift_report(reference_path: Path) -> str | None:
    """
    Generate an Evidently drift report and save as JSON.
    Returns the JSON file path.
    """
    if not LOG_FILE.exists():
        return None

    # Load datasets
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(LOG_FILE)

    # Drop timestamp/time columns
    for col in ["timestamp", "time"]:
        if col in reference_data.columns:
            reference_data = reference_data.drop(columns=[col])
        if col in current_data.columns:
            current_data = current_data.drop(columns=[col])

    # Keep only columns present in both datasets
    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    # Convert numeric columns to numeric
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        reference_data[col] = pd.to_numeric(reference_data[col], errors="coerce")
        current_data[col] = pd.to_numeric(current_data[col], errors="coerce")

    # Convert non-numeric columns to string
    non_numeric_cols = [c for c in current_data.columns if c not in numeric_cols]
    for col in non_numeric_cols:
        reference_data[col] = reference_data[col].astype(str)
        current_data[col] = current_data[col].astype(str)

    # Drop fully empty columns
    non_empty_cols = current_data.columns[~current_data.isna().all()].tolist()
    reference_data = reference_data[non_empty_cols]
    current_data = current_data[non_empty_cols]

    report = Report([
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,

    )
    logger.info(result)

    return result.json()