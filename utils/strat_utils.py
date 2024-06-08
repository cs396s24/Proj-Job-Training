import numpy as np
import pandas as pd


def calculate_stratified_effects(df, prop_col, treatment_col, outcome_col, num_strata):
    """
    Calculate the causal effect of the treatment on the outcome using stratification by propensity scores.

    Parameters:
    - df (pd.DataFrame): The dataset containing the propensity score, treatment, and outcome columns.
    - prop_col (str): The name of the propensity score column.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.
    - num_strata (int): The number of strata to divide the data into based on the propensity score.

    Returns:
    - overall_effect (float): The estimated causal effect of the treatment on the outcome.
    """
    df["strata"] = pd.qcut(df[prop_col], num_strata, labels=False)

    strata_effects = []
    strata_counts = []

    for stratum in range(num_strata):
        stratum_data = df[df["strata"] == stratum]
        treated = stratum_data[stratum_data[treatment_col] == 1]
        control = stratum_data[stratum_data[treatment_col] == 0]

        if len(treated) > 0 and len(control) > 0:
            treated_outcome = treated[outcome_col].mean()
            control_outcome = control[outcome_col].mean()
            effect = treated_outcome - control_outcome
            strata_effects.append(effect)
            strata_counts.append(len(treated))

    overall_effect = np.average(strata_effects, weights=strata_counts)
    return overall_effect


def calculate_grouped_effects(df, stratify_cols, treatment_col, outcome_col):
    """
    Calculate the causal effect of the treatment on the outcome using stratification by specified columns.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment, outcome, and stratification columns.
    - stratify_cols (list of str): A list of column names to stratify the data by.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.

    Returns:
    - overall_effect (float): The estimated causal effect of the treatment on the outcome.
    """
    grouped = df.groupby(stratify_cols)

    strata_effects = []
    strata_counts = []

    for _, group in grouped:
        treated = group[group[treatment_col] == 1]
        control = group[group[treatment_col] == 0]

        if len(treated) > 0 and len(control) > 0:
            treated_outcome = treated[outcome_col].mean()
            control_outcome = control[outcome_col].mean()
            effect = treated_outcome - control_outcome
            strata_effects.append(effect)
            strata_counts.append(len(treated))

    overall_effect = np.average(strata_effects, weights=strata_counts)
    return overall_effect
