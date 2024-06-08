import pandas as pd
import numpy as np


def calculate_stratified_effects(
    df, prop_col, treatment_col, outcome_col, num_strata
):
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
