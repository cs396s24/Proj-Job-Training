# Research Notes

## Evaluating the Econometric Evaluations of Training Programs with Experimental Data
(Robert J. LaLonde, 1986)

### Intro

* evaluate econometric methods is to compare them against experimentally determined results
* The paper compares the results from a field experiment, where individuals were randomly assigned to participate in a training program, against the array of estimates that an econometrician without experimental data might have produced.
* National Supported Work Demonstration (NSW): temporary employment program designed to help **disadvantaged workers lacking basic job skills** move into the labor market by giving them work experience and counseling in a sheltered environment.
*  Those assigned to the treatment group received all the benefits of the NSW program, while those assigned to the control group were left to fend for themselves.
* ex-drug addicts, ex-criminal offenders, and high school dropouts.
* For those assigned to the treatment group, the program guaranteed a job for 9 to 18 months, depending on the target group and site:
  * fat-fingered intervention
  * promised to get a job (incentive) and job training = better mental state (positive)
  * and were paid (albeit minimal) = ability to negotiate pay
  * vs no training = aimless + no revenue
* drop offs: allowed earnings to increase for satisfactory performance and attendance
* trainees could stay on their supported work jobs until their terms in the program expired and they were forced to  find regular employment.
* agencies at each site provided the treatment group members with different work experiences (but we're assuming everyone receives the same treatment)
  * e.g. trainees in Hartford worked at a gas station, while others worked at a printing shop
* male and female participants frequently performed different sorts of work.
  * The female participants usually worked in service occupations, whereas the male participants tended to work in construction occupation.
* The program cost$9,100 per AFDC participant and approximately$6,800 for the other target groups' trainees
* MDRC collected earnings and demographic data from both the treatment and control group members at the baseline (when the participants are randomly assigned) and every night months thereafter.
* Four baseline interviews:
  * many participants failed to complete these interviews
  * sample attrition biases the experimental results
  * fortunately, largest source of attriition doesn't affect the integrity of the experimental design
* All earnings presented are in 1982 dollars (accounts for inflation) -- inflation is a confounder.

### Nonexperimental Estimates

* In a dynamic economy, the trainees' earnings may grow even without an effective program
* Criticism that pre and post-earnings are a poor estimate of training effect
* the trainees' earnings without the program.
* Without experimental data, researchers estimate the earnings of the trainees by using the regression-adjusted earnings of a comparison group drawn from the population.
  * This adjustment takes into account that the observable characteristics of the trainees and the comparison group members differ their unobservable characteristics may differ as well
  * can we use backdoor estimates here? how do we measure the causal effect from observable data?

Model of earnings and program participation in the context of evaluating the effects of a training program. These equations represent a typical econometric approach to understanding how participation in a program (like training) influences earnings, while accounting for selection into the program.

#### Equation (1): Earnings Equation

$Y_{it} = D_i + \beta X_{it} + b_i + \nu_t + \epsilon_{it}$

- **$Y_{it}$**: Earnings of individual $i$ at time $t$.
- **$D_i$**: A binary indicator variable that equals 1 if the individual $i$ participated in the training program, and 0 otherwise. This coefficient measures the impact of the training program on earnings.
- **$\beta X_{it}$**: A vector of observable characteristics (such as age, education, experience) and their associated coefficients ($\beta$).
- **$b_i$**: An individual-specific effect that captures unobserved heterogeneity across individuals (e.g., innate ability, motivation).
- **$\nu_t$**: A time-specific effect that captures common shocks or trends affecting all individuals (e.g., macroeconomic conditions).
- **$\epsilon_{it}$**: The error term, which captures random noise or idiosyncratic factors affecting individual $i$'s earnings at time $t$.

#### Equation (2): Autoregressive Component of Earnings

$E_{it} - P_{it-1} = \eta_t$

- **$E_{it}$**: Expected earnings of individual $i$ at time $t$.
- **$P_{it-1}$**: Lagged earnings or some measure of past earnings of individual $i$.
- **$\eta_t$**: Time-specific error term that could be capturing the residual change or shock to earnings.

This equation suggests that the expected earnings are influenced by past earnings, introducing an autoregressive component.

#### Equation (3): Latent Variable for Program Participation

This is just propensity measured via logistic regression.

$d_{is} = \gamma Y_{is} + \delta Z_{is} + \tau_{is}$

- **$d_{is}$**: Latent variable representing the propensity or underlying tendency for individual $i$ to participate in the program at time $s$.
- **$Y_{is}$**: Earnings (or some measure related to earnings) of individual $i$ at time $s$.
- **$Z_{is}$**: A vector of other factors or characteristics influencing the decision to participate in the program (e.g., demographics, past job performance).
- **$\gamma$** and **$\delta$**: Coefficients measuring the influence of earnings and other factors on the participation decision.
- **$\tau_{is}$**: Error term capturing unobserved factors affecting the participation decision.

#### Equation (4): Program Participation Decision

$D_i = 1 \text{ if } d_{is} > 0; \quad D_i = 0 \text{ if } d_{is} \leq 0$

- **$D_i$**: Observed binary indicator of program participation for individual $i$.
- The decision to participate in the program is based on the latent variable $d_{is}$. If $d_{is}$ exceeds a threshold (typically zero), the individual participates in the program ($D_i = 1$); otherwise, they do not ($D_i = 0$).

### Questions

* How do we show that revenue is an unbiased estimator?
* How do we show that the randomized studies are fair comparsions?
  * Can we show distribution of confounders/controls?
  * Also check if the pre-training earnings are similar.
  * run a regression analysis to check where pre-treatment earnings are regressed on a treatment group indicator variable and possibly other covariates.
  * The key point is to see if the coefficient on the treatment group indicator is statistically significant. If it’s not, this indicates no significant difference between the groups’ pre-treatment earnings.
  * the coefficient of the treatment needs to be 0 if the initial revenue is an unbiased estimator.
  * we should also run a t-test on the parameter ($\beta_0$) of A to check if$\beta_0 = 0$is statistically significant.
  * to get p-value, use the package. More information on its workings can be found here: https://stats.stackexchange.com/questions/485768/how-is-a-p-value-computed-for-regression-coefficients-and-why-does-it-remain-un
* Can we use some form of matching or IPW for comparsion? Can we condition on a group and then compare results?
