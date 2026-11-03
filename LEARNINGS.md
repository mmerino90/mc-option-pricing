# What I learned building a Monte Carlo option pricer

I built this project to prepare for *Numerical Methods for Finance*, *Financial Engineering* and *Applied Computational Finance*. I started knowing basic probability (expectations, variance, the central limit theorem) and no stochastic calculus. These are the ideas that stuck, each tied to the experiment that taught it to me.

---

## 1. A price is an expectation, just not under the real-world probabilities

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\big[\text{payoff}\big]
$$

The surprising part is $\mathbb{Q}$. Because an option can be replicated with the stock and cash, its price can't depend on how fast *I* think the stock will grow, so it is computed in a fictional world where everything grows at the risk-free rate. That is why the simulation's drift is $r$, not an expected return.

The $-\tfrac{1}{2}\sigma^2$ in the geometric Brownian motion stopped looking like magic once I saw it as Jensen's inequality: $\mathbb{E}[e^X] = e^{m + v/2}$, so the drift has to be lowered by half the variance for the *average* price to grow at $r$.

## 2. The standard error is the real output

Monte Carlo's error falls like $1/\sqrt{N}$ (I measured a log-log slope of **−0.511**): **10× more accuracy costs 100× more simulations.** That rate doesn't depend on the dimension of the problem, which is why Monte Carlo wins when there are many variables.

What changed how I think about it: the estimate's **own** standard error matched the true error (95% intervals covered the Black–Scholes price **94.4%** of the time). For the exotic options later on there was no formula to compare against. The standard error was the only thing telling me whether to trust a number.

## 3. Reduce variance by using what you already know

Every variance reduction technique works by exploiting structure that plain Monte Carlo ignores, and the gain depends on how well that structure matches the payoff:

- **Antithetic variates** use the symmetry of the normal distribution. 2× at the money, 24× deep in the money, almost nothing far out of the money.
- **A control variate** is a linear regression in disguise: subtract the part of the payoff explained by something whose mean you know. With the stock as control: 7×. With the geometric Asian option, whose payoff is 0.9996-correlated with the arithmetic one: **1,800×**. The quality of the control is everything.
- **Sobol sequences** change the convergence rate itself, to roughly $1/N$, but only in low dimension.

I also learned to compare methods by **variance × time**, not variance alone: a method that halves the variance but triples the run time is a loss.

## 4. Finite differences without common random numbers are useless

Bumping a parameter and repricing with *fresh* random numbers gives an error proportional to $1/h$. With $h = 0.001$, the delta estimate was off by about 106 for a true delta of 0.64. Reusing the *same* random numbers makes the noise cancel, and the error goes flat at exactly the pathwise estimator's level. Pushed to $h \to 0$, finite differences with common random numbers *are* the pathwise method. The pathwise method breaks for discontinuous payoffs, though: a digital option would get a delta of exactly zero.

## 5. Two kinds of error that look the same

Checking a barrier only on discrete dates misses crossings in between. With 1,000 monitoring dates, the price was still **3 standard errors** too high, and no number of extra paths would fix that, because it is a **bias**, not noise. A tight confidence interval around a biased number is still wrong. The Brownian-bridge correction removed the bias with only 4 dates, which taught me that a little maths can beat a lot of compute.

## 6. Early exercise is a regression problem

Pricing an American option means deciding at every date whether to exercise, which requires the expected value of continuing, a **conditional expectation**. Longstaff–Schwartz estimates it with least squares on the simulated paths. This is where finance met machine learning for me:

- **Train/test split.** Fitting the exercise rule and pricing on the same paths lets the rule peek at the future. On independent paths the price is a slight *under*-estimate, which is the honest direction to be wrong in.
- **Underfitting has a price.** A linear basis mispriced the option by 0.06 (six standard errors); a cubic basis matched the binomial tree to within 0.0004.

## 7. Measure before you optimise

I expected C++ to be dramatically faster. Single-threaded, it was only **1.1–1.5×** faster than vectorised NumPy, because NumPy already runs C. Against pure Python it was **39–86×** faster. The real wins came from things NumPy can't do:
- stop a path the moment it hits the barrier;
- avoid allocating $N \times m$ arrays;
- spread the paths across cores, each thread with its own random generator.

Twelve threads gave 2.5–5× over one thread, not 12×. The laptop chip mixes 2 performance cores with 8 efficiency cores, which was a concrete lesson in why hardware matters as much as code.

## 8. Habits I'll keep

- **Always have a ground truth:** a closed form, a binomial tree, or a limiting case (a barrier at zero is a vanilla call; one Asian fixing is a European option).
- **Write statistical tests:** "within 3 standard errors", not "equals".
- **Fix the seeds, and know what changing them means.** Shared seeds make errors correlated, which can mislead you or be exploited on purpose.

---

## What I'd do next

- Calibrate a **Heston** stochastic-volatility model to real option prices, where Euler discretisation error becomes a real issue.
- **Likelihood-ratio Greeks**, to handle the discontinuous payoffs where pathwise fails.
- **Quasi-Monte Carlo for path-dependent options**, using a Brownian-bridge construction so that Sobol's best dimensions drive the most important parts of each path.
