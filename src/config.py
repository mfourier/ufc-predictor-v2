"""
Configuration module for the UFC Fight Predictor project.

This file centralizes all model-related configuration for the machine learning pipeline, 
including:

- Definitions of supported scikit-learn classifiers used for fight outcome prediction.
- Default hyperparameter grids for each model, structured for compatibility with GridSearchCV.
- Dictionaries mapping model classes and identifiers to human-readable names ("pretty names").
- Standardized filenames for saving/loading trained models.
- ANSI color codes for enhanced terminal output styling.

Key structures:

- default_params: 
    Dictionary mapping display names to tuples of (estimator, hyperparameter grid). 
    Used for automatic selection and hyperparameter optimization of models. 
    Each grid is tuned for balanced search space coverage and computational feasibility.

- pretty_names: 
    Maps scikit-learn class names to display-friendly names for reporting and visualization.

- pretty_model_name: 
    Maps internal model file identifiers (e.g., "rf_best") to display names, ensuring consistency 
    in file naming and result presentation.

- file_model_name: 
    Maps display names back to standardized file identifiers for model persistence.

- colors: 
    ANSI color codes for consistent and readable CLI output across scripts.

This design enforces a single source of truth for model-related settings, reducing duplication 
and maintenance overhead throughout the codebase.

Any additions or changes to model support should be made in this file to ensure consistent 
integration with the rest of the project.

Author: [Maximiliano Lioi, MSc. Mathematics]
"""


# Extended ANSI color codes for console outputs
colors = {
    "default": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "reverse": "\033[7m"
}
