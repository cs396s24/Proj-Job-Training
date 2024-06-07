import numpy as np
import statsmodels.formula.api as smf


def ipw(df, treatment_col, outcome_col, confounder_cols):
    df_copy = df.copy()
    results = []
    treatments = sorted(np.unique(df_copy[treatment_col]))

    # Fitting propensity model
    formula = f"{treatment_col} ~ " + " + ".join(confounder_cols)
    model = smf.mnlogit(formula=formula, data=df_copy).fit(disp=False)

    # Calculating IPW
    propensity_scores = model.predict(df_copy).reset_index(drop=True)
    df_copy["propensity"] = df_copy.apply(
        lambda row: propensity_scores.loc[row.name, int(row[treatment_col])], axis=1
    )
    df_copy["ipw"] = 1 / df_copy["propensity"]

    # Calculate E[Y^a] for each unique a
    for a_val in treatments:
        df_a = df_copy.loc[df_copy[treatment_col] == a_val, :].reset_index(drop=True)
        weighted_mean = np.average(df_a[outcome_col], weights=df_a["ipw"])
        results.append(weighted_mean)

    return results[1] - results[0]


def ipw_probs(df, treatment_col, outcome_col, confounder_cols):
    df_copy = df.copy()
    results = []
    treatments = sorted(np.unique(df_copy[treatment_col]))

    # Fitting propensity model
    formula = f"{treatment_col} ~ " + " + ".join(confounder_cols)
    model = smf.mnlogit(formula=formula, data=df_copy).fit(disp=False)

    # Calculating IPW
    propensity_scores = model.predict(df_copy).reset_index(drop=True)
    df_copy["propensity"] = df_copy.apply(
        lambda row: propensity_scores.loc[row.name, int(row[treatment_col])], axis=1
    )
    df_copy["ipw"] = 1 / df_copy["propensity"]

    return df_copy[["id", "ipw"]]
