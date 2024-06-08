import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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


def backdoor_classifier(
    df, treatment_col, outcome_col, confounder_cols, classifier_name="DecisionTree"
):
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

        # Prepare the data
        X = df_a[confounder_cols]
        y = df_a[outcome_col]

        # Fit the classifier model
        model = classifier.fit(X, y)

        # Predict probabilities on the entire dataset
        X_full = df[confounder_cols]
        y_pred_prob = model.predict_proba(X_full)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        results.append(np.mean(y_pred))

    return results[1] / results[0]
