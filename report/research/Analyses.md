# Analyses - 1

## Sensitivity Analysis

https://docs.doubleml.org/tutorial/stable/notebooks/sensitivity_lalonde.html

* illustrate the use of sensitivity analysis on the Lalonde dataset by comparing experimental and observational data
* Assesses how sensitive the estimated treatment effects are to unobserved confounding.
* We might cover this in class.

## TDS: Propensity & Reputation

https://towardsdatascience.com/methods-for-inferring-causality-52e4144f6865

* Calculates propensity to match on instead of covariates.
* We can use propensity with the backdoor adjustment formula to calculate the causal estimate.
* Unique idea: Try different algorithms for propensity that are not logistic regression. A decision tree might work better here and be more interpretable.

## IPW with DoWhy

https://petergtz.github.io/dowhy/main/example_notebooks/dowhy_lalonde_example.html

* This applies DoWhy to calculate IPW and a bunch of other causal things.

## Average Treatment Effect

https://rugg2.github.io/Lalonde%20dataset%20-%20Causal%20Inference.html

* $E[Y_1-Y_0]$
* Measured in a randomized trial. Ignorability assumed.
* Also uses trim samples for propensity scores -- what is this?
* Stratifies and measures causal effect.
* Blocking estimator -- what is this?
* Again uses matching or IPW.
* "By glancing at it, Lalonde seemed to know the gender of participants, which does not seem to be in this dataset, or may be hidden in the NSW vs AFDC."

## Lalonde example

https://lendle.github.io/TargetedLearning.jl/user-guide/lalonde_example/

* We're not using the LaLonde dataset, we're using the "Dehejia-Wahha Sample".
* TMLE, CTMLE for the average treatment effect.

## Causallib example

https://github.com/BiomedSciAI/causallib/blob/master/examples/lalonde.ipynb

* Economists have long-hypothesized that training programs could improve the labor market prospects of participants. In an attempt to test (or demonstrate) this, the National Supported Work (NSW) Demonstration was initiated using combined private and federal funding. This program was implemented between 1975 and 1979 in 15 locations across the US. The program provided 6-18 month training for individuals who had faced economic and social problems (such as women receiving Aid to Families with Dependent Children, former drug addicts, ex-convicts, and former juvenile delinquents, etc.).
* Participants were randomly assigned into experimental group (Support Work Programs) and control groups. However, due to the long duration of the study, participants joining the program at the beginning had different characteristics than people joining later.
Therefore, this covariate shift should be adjusted for in order to estimate the true causal effect of the job-program on future employment.
* This is probably the entire dataset.

## Lalonde Sample

https://github.com/wayfair/pylift/blob/master/examples/Lalonde/Lalonde_sample.ipynb

* These folks just use pylift.
* What is pylift?

## CI on Lalonde Dataset

https://github.com/paul-english/causal-inference-notes/blob/master/Causal%20Inference%20on%20the%20Lalonde%20Dataset.ipynb

* They just use the causalinference library.
* Also estimate the counterfactual observations

## Matching

https://ose-data-science.readthedocs.io/en/latest/problem-sets/matching-estimators/notebook.html

* They perform matching based on propensity scores.

## Causal Statistics

https://rpubs.com/omesner/causal

* Little to no work on the LaLonde dataset.
