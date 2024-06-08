import numpy as np
import pandas as pd


def marriage_prob(x):
    """
    Calculate the probability of marriage based on age using a fitted saturating exponential function.

    Parameters:
    x (int or float): Age of the individual.

    Returns:
    float: Probability of marriage.
    """
    return 0.9 * (1 - np.exp(-0.25 * (x - 16)))


def compute_group_stats(df, group_vars, re_vars):
    """
    Calculate means and covariances for each group in the dataframe.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    group_vars (list of str): List of column names to group by.
    re_vars (list of str): List of column names for which to calculate means and covariances.

    Returns:
    dict: A dictionary where keys are group identifiers and values are tuples of (means, covariances).
    """
    group_stats = {}
    grouped = df.groupby(group_vars)
    for group, data in grouped:
        means = data[re_vars].mean().values
        cov = data[re_vars].cov().values
        group_stats[group] = (means, cov)
    return group_stats


def generate_synthetic_data(n, prob_black, prob_hispanic, prob_white, group_stats):
    """
    Generate synthetic data based on group statistics and probabilities for different ethnicities.

    Parameters:
    n (int): Number of samples to generate.
    prob_black (float): Probability of an individual being black.
    prob_hispanic (float): Probability of an individual being Hispanic.
    prob_white (float): Probability of an individual being white.
    group_stats (dict): Dictionary containing group statistics (means and covariances).

    Returns:
    pd.DataFrame: DataFrame containing the generated synthetic data with columns:
                      ['black', 'hispanic', 'married', 'education', 'treat', 'age', 're74', 're75', 're78'].
    """
    data = []

    while len(data) < n:
        ethnicity = np.random.choice(
            ["black", "hispanic", "white"], p=[prob_black, prob_hispanic, prob_white]
        )

        black = int(ethnicity == "black")
        hispanic = int(ethnicity == "hispanic")

        # Generate multivariate normal data
        group = (black, hispanic)
        means, cov = group_stats[group]
        sample = np.random.multivariate_normal(means, cov)

        age = sample[0].round().astype(int)
        education = sample[1].round().astype(int)
        if education < 0 or age < 16:
            continue
        p = 0.25 + (education >= 12) * 0.25 + black * 0.25
        treat = np.random.binomial(1, p)

        # Generate married based on age using the fitted saturating exponential function
        married = np.random.binomial(1, marriage_prob(age))

        re74, re75, re78 = np.clip(sample[2:], 0, None)

        # Adjust re78 with the treatment effect if treated
        re78 += treat * np.random.normal(5000, 500)

        # Append valid data
        data.append([black, hispanic, married, education, treat, age, re74, re75, re78])

    # Create DataFrame from valid data
    synthetic_data = pd.DataFrame(
        data,
        columns=[
            "black",
            "hispanic",
            "married",
            "education",
            "treat",
            "age",
            "re74",
            "re75",
            "re78",
        ],
    )

    return synthetic_data
