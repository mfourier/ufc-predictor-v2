# Author: Maximiliano Lioi | License: MIT

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_training_result(
    model_name: str,
    best_params: dict,
    metrics: dict,
    duration: float,
    log_path: str = "../data/results/training_log_v2.csv"
) -> None:
    """
    Log the training results of a model into a cumulative CSV file.

    Args:
        model_name (str): Name of the model used in training.
        best_params (dict): Dictionary of hyperparameters found by GridSearchCV.
        metrics (dict): Dictionary containing evaluation metrics (accuracy, F1, etc.).
        duration (float): Duration of training in seconds.
        log_path (str): Path where the CSV log will be stored.

    Returns:
        None
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'duration_sec': round(duration, 2),
        **metrics,
        **{f'param_{k}': v for k, v in best_params.items()}
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_path, index=False)
    logger.info(f"✅ Training logged to {log_path}")
