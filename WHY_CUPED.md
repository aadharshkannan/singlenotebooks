## Why not just “difference of means with IID draws”?

Mean‑difference t‑tests assume i.i.d. outcomes. In our setting those assumptions break:

* **Same test cases across arms → strong within‑case correlation.** Ignoring it wastes signal and misstates SEs. The effective sample size is the **# of unique test cases**, not the # of judgments.
* **LLM judges introduce heteroskedastic, autocorrelated noise** (prompt/order effects, drift).
* **Case mix dominates variance.** Some cases are intrinsically easy/hard; that variance swamps our treatment effect if we don’t adjust.

**CUPED (and its Bayesian analogue)** deliberately subtracts the predictable part of each outcome using pre‑treatment information (our **PREVIOUS_CODE_PATH** performance), yielding **much lower variance** but the **same unbiased effect**.

---

## CUPED (for rubric scores)

**Idea.** Use the prior score of each test case as a **control variate** to remove per‑case difficulty from the NEW vs PREVIOUS comparison.

**Notation (per test case i):**

* (s^{\text{prev}}_i): rubric score under **PREVIOUS_CODE_PATH** (0–1 or 0–100).
* (s^{\text{new}}_i): rubric score under **NEW_CODE_PATH**.
* We care about (\Delta = \mathbb{E}[s^{\text{new}} - s^{\text{prev}}]).

**Paired CUPED estimator (recommended for our “same cases in both arms” setup):**

1. Compute the per‑case difference (D_i = s^{\text{new}}_i - s^{\text{prev}}_i).
2. Center the baseline (Z_i = s^{\text{prev}}_i - \bar{s}^{\text{prev}}).
3. Estimate
   [
   \hat{\theta} ;=; \frac{\operatorname{Cov}(D, Z)}{\operatorname{Var}(Z)} \quad\text{(use K‑fold cross‑fitting or estimate on a holdout of PREVIOUS runs to be safe).}
   ]
4. Adjust the differences (D_i^{*} = D_i - \hat{\theta} Z_i).
5. Report the effect (\hat{\Delta} = \overline{D^{*}}).
   A simple SE is ( \operatorname{sd}(D^{*})/\sqrt{N_{\text{cases}}}). Prefer a **case‑level bootstrap** if judges/runs are clustered.

**Why this helps.** With the optimal (\theta),
[
\operatorname{Var}(D^{*}) = \operatorname{Var}(D),\bigl(1-\rho_{D,Z}^{2}\bigr),
]
so variance shrinks by the squared correlation between the **baseline** and the **improvement**. If (\rho_{D,Z}=0.8), variance drops **64%** (≈ **2.8×** effective sample size).

> ⚠️ Classic “two‑arm CUPED” (subtracting (\theta(X-\bar X)) from **both** arms) gives little/no benefit when the same cases appear in both arms because the adjustment cancels in the mean difference. The **paired** version above is the right tool here.

**Minimal pseudocode**

```python
# s_prev, s_new are arrays aligned by test case
D = s_new - s_prev
Z = s_prev - s_prev.mean()

# cross-fit theta if you want to be purist; simple estimate works well in practice
theta = np.cov(D, Z, ddof=1)[0,1] / np.var(Z, ddof=1)

D_star = D - theta * Z
delta_hat = D_star.mean()
se = D_star.std(ddof=1) / np.sqrt(len(D_star))
```

**Reporting checklist**

* (\hat{\Delta}) with 95% CI;
* **Variance reduction** (1 - \operatorname{Var}(D^{*})/\operatorname{Var}(D));
* **Correlation** (\rho_{D,Z});
* Effective sample size gain ( \text{ESS}*{\text{gain}} \approx 1/(1-\rho*{D,Z}^{2})).

---

## Bayesian CUPED (for assertion pass/fail)

When outcomes are binary (LLM **assertions**), use a **hierarchical Bayesian model** that plays the same control‑variate role as CUPED while handling discreteness, small counts, and judge/run effects.

**Data (per case i):**

* (y^{\text{prev}}_i \sim \text{Binomial}(m_i, p^{\text{prev}}_i)) and
  (y^{\text{new}}_i \sim \text{Binomial}(m_i, p^{\text{new}}_i)), where (m_i) is #assertions.
* Define baseline covariate (x_i = y^{\text{prev}}_i/m_i) (prev pass rate).

**Model (logistic hierarchical “Bayesian CUPED”):**
[
\begin{aligned}
y_{i,a} &\sim \text{Binomial}(m_{i,a},, p_{i,a}) \quad a\in{\text{prev},\text{new}},\
\text{logit}, p_{i,a} &= \alpha ;+; \delta\cdot\mathbf{1}[a=\text{new}] ;+; \beta,(x_i - \bar{x}) \
&\quad +, u_{\text{case}(i)} + u_{\text{judge_run}(i)} \quad\text{with } u\sim\mathcal{N}(0,\sigma^2).
\end{aligned}
]

* (\delta) is the **treatment effect** on log‑odds (what we care about).
* (\beta) plays the **CUPED control‑variate** role, borrowing strength from (x_i).
* Random effects soak up non‑IID noise (case difficulty, judge/run drift).

**Outputs to report**

* ( \mathbb{P}(\delta>0 \mid \text{data}) ) (probability NEW improves).
* Posterior mean of absolute lift at a baseline pass rate (p_0):
  (\Delta p \approx \sigma(\text{logit}(p_0)+\delta) - p_0).
* Posterior predictive pass rate difference across the test set.
* Effective sample size gain (e.g., via posterior variance ratio vs model without (x_i)).

**Why this helps.** The model **shrinks** noisy assertion counts toward what we already know from **PREVIOUS_CODE_PATH**, dramatically narrowing uncertainty—especially when many cases have few assertions or rare failures.

---

## Practical guidance & guardrails

* **Standardize** rubric scales (e.g., to 0–1) and keep judge prompts identical across arms. Randomize evaluation **order** to avoid drift.
* **Cross‑fitting (\theta)**: split cases into K folds; estimate (\theta) on K−1 folds, apply to the held‑out fold; concatenate. This avoids any small finite‑sample bias debates.
* **Clustered uncertainty**: bootstrap or cluster‑robust SEs at the **test‑case** level (and judge‑run if applicable).
* **Diagnostics**: report (\rho_{D,Z}) (CUPED usefulness), posterior R‑hat/ESS (Bayesian), and sanity‑check that effects are not driven by a few outlier cases.
* **Interpretability**: For rubrics, also show % of cases improved/worse; for assertions, show per‑assertion lift distribution.

---

### TL;DR

* **CUPED** on rubric scores = *paired control‑variate on the per‑case delta*: same point estimate as mean difference, **much smaller variance**.
* **Bayesian CUPED** on assertions = *logistic hierarchical model with the previous pass rate as a covariate*: principled uncertainty with the same variance‑reduction idea.
* Both are **strictly better than naive IID mean differences** for our “same test cases, LLM judges” scenario.