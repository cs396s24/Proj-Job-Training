import numpy as np
import statsmodels.formula.api as smf


def ipw(df, treatment_col, outcome_col, confounder_cols):
    """
    Estimate the causal effect of the treatment on the outcome using Inverse
    Probability Weighting (IPW).

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment, outcome, and
    confounder columns.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.
    - confounder_cols (list of str): A list of names of confounder columns.

    Returns:
    - effect (float): The estimated causal effect of the treatment on the outcome.
    """
    df_copy = df.copy()
    results = []
    treatments = sorted(np.unique(df_copy[treatment_col]))

    formula = f"{treatment_col} ~ " + " + ".join(confounder_cols)
    model = smf.mnlogit(formula=formula, data=df_copy).fit(disp=False)

    propensity_scores = model.predict(df_copy).reset_index(drop=True)
    df_copy["propensity"] = df_copy.apply(
        lambda row: propensity_scores.loc[row.name, int(row[treatment_col])], axis=1
    )
    df_copy["ipw"] = 1 / df_copy["propensity"]

    for a_val in treatments:
        df_a = df_copy.loc[df_copy[treatment_col] == a_val, :].reset_index(drop=True)
        weighted_mean = np.average(df_a[outcome_col], weights=df_a["ipw"])
        results.append(weighted_mean)

    return results[1] - results[0]


def prop_probs(df, treatment_col, confounder_cols):
    """
    Calculate propensity scores for each observation in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment and confounder
    columns.
    - treatment_col (str): The name of the treatment column.
    - confounder_cols (list of str): A list of names of confounder columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the original data with an additional
    column for propensity scores.
    """
    df_copy = df.copy()

    formula = f"{treatment_col} ~ " + " + ".join(confounder_cols)
    model = smf.mnlogit(formula=formula, data=df_copy).fit(disp=False)

    propensity_scores = model.predict(df_copy).reset_index(drop=True)
    df_copy["propensity"] = df_copy.apply(
        lambda row: propensity_scores.loc[row.name, int(row[treatment_col])], axis=1
    )
    return df_copy[["id", "propensity"]]


def ipw_probs(df, treatment_col, confounder_cols):
    """
    Calculate inverse probability weights (IPW) for each observation in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment and confounder
    columns.
    - treatment_col (str): The name of the treatment column.
    - confounder_cols (list of str): A list of names of confounder columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the original data with an additional
    column for IPW.
    """
    df_copy = prop_probs(df, treatment_col, confounder_cols)
    df_copy["ipw"] = 1 / df_copy["propensity"]

    return df_copy[["id", "ipw"]]
