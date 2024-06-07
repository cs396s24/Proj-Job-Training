import numpy as np
import statsmodels.formula.api as smf


def backdoor_param_a(df, treatment_col, outcome_col, confounder_cols):
    results = []

    formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(confounder_cols)
    model = smf.ols(formula=formula, data=df).fit()
    param_a = model.params[treatment_col]

    return param_a


def mean_diff_estimator(df, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    effect = treated[outcome_col].mean() - control[outcome_col].mean()
    return effect


def backdoor_lr(df, treatment_col, outcome_col, confounder_cols):
    results = []
    treatments = sorted(np.unique(df[treatment_col]))

    # predicting E[Y^a] for each unique a
    for a_val in treatments:
        df_a = df.loc[df[treatment_col] == a_val, :].reset_index(drop=True)
        formula = f"{outcome_col} ~ " + " + ".join(confounder_cols)
        model = smf.ols(formula=formula, data=df_a).fit()

        # predict using the entire dataset
        y_pred = model.predict(df)
        results.append(np.mean(y_pred))

    return results[1] - results[0]
