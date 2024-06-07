import numpy as np
from tqdm import tqdm


def bootstrap(df, function, n=50, ci=95, set_seed=True, **kwargs):
    if set_seed:
        np.random.seed(42)
    results = []

    for _ in range(n):
        # this will ensure our bootstrapped sample is the same length
        # with replacement: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        new_df = df.sample(n=df.shape[0], replace=True)
        assert new_df.shape == df.shape  # same shape
        results.append(function(new_df, **kwargs))

    # CI calculations: https://statisticsbyjim.com/hypothesis-testing/bootstrapping/
    # E.g. 95% CI is 2.5% - 97.5%
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    results_arr = np.array(results)
    lower_ci = np.percentile(results_arr, lower_percentile, axis=0)
    upper_ci = np.percentile(results_arr, upper_percentile, axis=0)
    mean_ = np.mean(results_arr)

    return np.array([mean_, lower_ci, upper_ci])


def bootstrap_experiment(df, function, num_exp=10, n=50, ci=95, **kwargs):
    experiment_results = []

    for i in tqdm(range(num_exp), desc=f"Running experiments"):
        ci_result = bootstrap(df, function, n=n, ci=ci, set_seed=False, **kwargs)
        experiment_results.append(ci_result)

    return np.array(experiment_results)
