import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def backdoor_param_a(df, treatment_col, outcome_col, confounder_cols):
    """
    Estimate the causal effect of the treatment on the outcome using the backdoor adjustment method.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment, outcome, and confounder columns.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.
    - confounder_cols (list of str): A list of names of confounder columns.

    Returns:
    - param_a (float): The estimated causal effect of the treatment on the outcome.
    """
    formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(confounder_cols)
    model = smf.ols(formula=formula, data=df).fit()
    param_a = model.params[treatment_col]

    return param_a


def mean_diff_estimator(df, treatment_col, outcome_col):
    """
    Estimate the causal effect of the treatment on the outcome using the mean difference estimator.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment and outcome columns.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.

    Returns:
    - effect (float): The estimated causal effect of the treatment on the outcome.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    effect = treated[outcome_col].mean() - control[outcome_col].mean()
    return effect


def backdoor_lr(df, treatment_col, outcome_col, confounder_cols):
    """
    Estimate the causal effect of the treatment on the outcome using a linear regression model
    for each treatment level and then averaging the predictions.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment, outcome, and confounder columns.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.
    - confounder_cols (list of str): A list of names of confounder columns.

    Returns:
    - effect (float): The estimated causal effect of the treatment on the outcome.
    """
    results = []
    treatments = sorted(np.unique(df[treatment_col]))

    for a_val in treatments:
        df_a = df.loc[df[treatment_col] == a_val, :].reset_index(drop=True)
        formula = f"{outcome_col} ~ " + " + ".join(confounder_cols)
        model = smf.ols(formula=formula, data=df_a).fit()
        y_pred = model.predict(df)
        results.append(np.mean(y_pred))

    return results[1] - results[0]


def backdoor_classifier(
    df, treatment_col, outcome_col, confounder_cols, classifier_name="DecisionTree"
):
    """
    Estimate the causal effect of the treatment on the outcome using a specified classifier.

    Parameters:
    - df (pd.DataFrame): The dataset containing the treatment, outcome, and confounder columns.
    - treatment_col (str): The name of the treatment column.
    - outcome_col (str): The name of the outcome column.
    - confounder_cols (list of str): A list of names of confounder columns.
    - classifier_name (str): The name of the classifier to use ('DecisionTree' or 'LogisticRegression').

    Returns:
    - effect (float): The estimated causal effect of the treatment on the outcome.
    """
    classifiers = {
        "DecisionTree": DecisionTreeClassifier,
        "LogisticRegression": LogisticRegression,
    }

    if classifier_name not in classifiers:
        raise ValueError(
            f"Classifier '{classifier_name}' is not supported. Choose from {list(classifiers.keys())}"
        )

    classifier = classifiers[classifier_name]()

    results = []
    treatments = sorted(np.unique(df[treatment_col]))

    for a_val in treatments:
        df_a = df.loc[df[treatment_col] == a_val, :].reset_index(drop=True)

        X = df_a[confounder_cols]
        y = df_a[outcome_col]

        model = classifier.fit(X, y)

        X_full = df[confounder_cols]
        y_pred_prob = model.predict_proba(X_full)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        results.append(np.mean(y_pred))

    return results[1] / results[0]
