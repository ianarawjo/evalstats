# Literature notes: CIs for a paired binary difference under clustering / repeated runs

Compiled 2026-08-25. Scope: what the statistics literature offers for a confidence
interval on the difference of two **paired** binary proportions when each item is
measured **R times** (clustered / over-dispersed matched-pair binary data), i.e. our
"two LLMs, n items, R seeded runs each, 0/1 per run" setting.

Mapping to the clinical-statistics vocabulary this literature uses:

| our setting | their setting |
|---|---|
| test item *i* | **cluster** *k* (patient, litter, eye-pair, study centre) |
| run *r* within item | **unit** within cluster (lesion, tooth, image) |
| model A vs model B | the two **procedures** / treatments in the matched pair |
| R runs per item | cluster size `n_k` (equal cluster sizes in our case) |
| item-level 2x2 counts | `a_k, b_k, c_k, d_k` (b = A right/B wrong, c = A wrong/B right) |

Everything below is stated at the level of abstracts, package documentation and R
source I could actually read. Points I could **not** verify from a primary source are
flagged explicitly in the last section — several key formula-level details are behind
paywalls.

---

## 1. Bottom line

- The clustered matched-pair problem **is** a recognised, named problem in the
  biostatistics literature, with a ~35-year lineage. We are not inventing it.
- There is a clear **testing** literature (five named tests, an R package) and a much
  thinner **interval-estimation** literature (essentially one paper).
- The closest thing to a "Fagerland-equivalent" systematic CI comparison for the
  clustered case is **Yang, Sun & Hardin (2012), *Pharmaceutical Statistics***. It is a
  single 8-page paper with ~6 citations, none of them methodological follow-ups. It is
  not remotely the settled, heavily-replicated consensus that Fagerland et al. (2014) is
  for the single-run case.
- Its headline recommendation is, in effect, **the design-effect approach**: "for small
  to medium numbers of clusters, the intracluster correlation coefficient-adjusted
  McNemar statistic and its associated Wald or Score CIs are preferred" (Yang, Sun &
  Hardin 2012, abstract). So an ICC-adjusted / design-effect interval **is** citable.
- Our specific implementation, however, is **not** the Eliasziw–Donner design effect. The
  `R_eff` step in `mj_floor_paired_ci_multirun_effective` is applied on top of a variance
  that is *already* cluster-level, and algebraically it either cancels exactly or inflates
  the variance (see §5). That part is ours, not the literature's.

---

## 2. Method catalogue

### 2.1 Eliasziw & Donner (1991) — ICC-adjusted McNemar. **This is the design-effect method.**

> Eliasziw, M. & Donner, A. (1991). Application of the McNemar test to non-independent
> matched pair data. *Statistics in Medicine* **10**(12), 1981–1991.
> DOI [10.1002/sim.4780101211](https://doi.org/10.1002/sim.4780101211). Cited ~114x.

- **Idea:** estimate the intra-cluster correlation among *discordant* pairs, then divide
  McNemar's chi-square by a correction factor.
- **Exact form** (read from the `clust.bin.pair` R source, `R/eliasziw.R`):
  `X2_di = X2_McNemar / C`, with `C = 1 + (n_c - 1) * rho_tilde`.
  That is **literally the Kish design effect** `1 + (m-1)*rho`, where
  `n_c = S0 + Kd*(S_bar - S0)` is an effective number of discordant responses
  (`Sk = b_k + c_k`, `Kd` = number of clusters with at least one discordant pair) and
  `rho_tilde` is a transformed ICC derived from an ANOVA-style estimator `rho_tilde*`
  (`BMS`/`WMS` pooled over the four cells) and the marginal discordance probabilities.
- **Assumes:** a Dirichlet-multinomial (beta-binomial-type) within-cluster distribution
  for the ICC estimator to be consistent.
- **Closed form:** yes, for the test. The paper is a *test*, not a CI.
- **Software:** `clust.bin.pair::clust.bin.pair(..., method = "eliasziw")` (CRAN, v0.1.2,
  2018, MIT). Tests/p-values only — the package returns an `htest` with statistic and
  p-value and **no** confidence interval.
- **Known defects** (per Wu 2019, below): the first ICC estimator is uncomputable when
  discordant pairs are few; the second is inconsistent when the Dirichlet-multinomial
  assumption fails.

**Downstream of it:**
- Gönen, M. (2004). Sample size and power for McNemar's test with clustered data.
  *Statistics in Medicine* **23**(14), 2283–2294. DOI
  [10.1002/sim.1768](https://doi.org/10.1002/sim.1768). Power/sample size for the
  *adjusted* (Eliasziw–Donner) McNemar, constant cluster size. Reported >10% error at
  high ICC and low discordance.
- Wu, Y. (2018). Power calculation of adjusted McNemar's test based on clustered data of
  varying cluster size. *Biometrical Journal* **60**(6), 1190–1200. DOI
  [10.1002/bimj.201800034](https://doi.org/10.1002/bimj.201800034). Extends Gönen to
  unequal cluster sizes; also gives a *more accurate* reduced power formula for the
  **fixed cluster size** case — which is ours.
- Wu, Y. (2019/2021). A robust adjustment to McNemar test when the data are clustered.
  *Communications in Statistics — Theory and Methods* **50**(6), 1515–1529. DOI
  [10.1080/03610926.2019.1651864](https://doi.org/10.1080/03610926.2019.1651864).
  Replaces Eliasziw–Donner's ICC with one estimable from **both** discordant and
  concordant pairs, consistent without the Dirichlet-multinomial assumption; abstract
  states size and power are comparable. Relevant to us because our discordance counts can
  be sparse.

### 2.2 Obuchowski (1998) — assumption-free cluster-level (sandwich-style) statistic

> Obuchowski, N. A. (1998). On the comparison of correlated proportions for clustered
> data. *Statistics in Medicine* **17**(13), 1495–1507. DOI
> [10.1002/(SICI)1097-0258(19980715)17:13<1495::AID-SIM863>3.0.CO;2-I](https://doi.org/10.1002/(SICI)1097-0258(19980715)17:13%3C1495::AID-SIM863%3E3.0.CO;2-I).
> Cited ~108x. PMID 9695194.

- **Statistic** (from `clust.bin.pair`, in Yang et al. 2010's notation):
  `X2 = ((K-1)/K) * (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2`.
  This is an **uncentered cluster-robust / empirical-sandwich** variance at the cluster
  level, with a `(K-1)/K` finite-sample factor. Valid under H0 because `E[b_k - c_k] = 0`.
- **Assumes:** essentially nothing about the within-cluster correlation structure — "The
  proposed method is simple to implement and makes no assumptions about the correlation
  structure" (abstract).
- **Simulation evidence in the paper itself:** compares size and power against
  Eliasziw–Donner. McNemar's size "can greatly exceed the nominal level" under ICC;
  Eliasziw–Donner is inflated for some correlation patterns; Obuchowski is close to
  nominal but slightly less powerful.
- **Closed form:** yes. **Software:** `clust.bin.pair(method = "obuchowski")`. Test only.

### 2.3 Durkalski, Palesch, Lipsitz & Rust (2003) — method-of-moments variance adjustment

> Durkalski, V. L., Palesch, Y. Y., Lipsitz, S. R. & Rust, P. F. (2003). Analysis of
> clustered matched-pair data. *Statistics in Medicine* **22**(15), 2417–2428. DOI
> [10.1002/sim.1438](https://doi.org/10.1002/sim.1438). PMID 12872299.

- **Statistic** (from `clust.bin.pair`):
  `X2 = (sum_k (b_k - c_k)/n_k)^2 / sum_k ((b_k - c_k)/n_k)^2`.
- **Note for us:** with **equal cluster sizes** (`n_k ≡ R`, exactly our case) this is
  identical to Obuchowski's statistic up to the `(K-1)/K` factor. Durkalski's whole point
  was robustness to *unequal* cluster sizes and heterogeneous success probabilities, which
  we do not have.
- **Assumes:** no distributional assumption, no correlation-structure assumption; MoM
  variance estimator.
- Companion non-inferiority paper: Durkalski et al. (2003), *Statistics in Medicine*
  **22**(2), 279–290, DOI [10.1002/sim.1385](https://doi.org/10.1002/sim.1385) — a
  Wald-type non-inferiority statistic.
- **Closed form:** yes. **Software:** `clust.bin.pair(method = "durkalski")`. Test only.

### 2.4 Yang, Sun & Hardin (2010) — modified Obuchowski test. **Recommended for equal cluster sizes.**

> Yang, Z., Sun, X. & Hardin, J. W. (2010). A note on the tests for clustered matched-pair
> binary data. *Biometrical Journal* **52**(5), 638–652. DOI
> [10.1002/bimj.201000035](https://doi.org/10.1002/bimj.201000035). Cited ~45x.

- **Statistic** (from `clust.bin.pair`, `R/yang.R`):
  `X2_mo = ((K-1)/K) * (sum_k (b_k-c_k))^2 / ( 0.5 * sum_k [ ((b_k-c_k) - n_k*(p1~ - p2~))^2 + (b_k-c_k)^2 ] )`
  with `p1~ = sum(b_k)/N`, `p2~ = sum(c_k)/N`. I.e. it averages the **centred** and
  **uncentred** cluster-level sums of squares — a hybrid between the null-variance and the
  empirical-variance sandwich.
- **Explicit recommendation from the abstract:** "(i) for **equal cluster size, the
  modified Obuchowski test is always preferred**; (ii) for varying cluster size Durkalski's
  test can be used for a small number of clusters (K < 50), whereas for K >= 50 the
  modified Obuchowski test is preferred." Obuchowski's original is "most conservative".
- **This is the single most directly on-point recommendation for us**: R runs per item is
  the equal-cluster-size case.
- **Closed form:** yes. **Software:** `clust.bin.pair(method = "yang")` — the package
  default. Test only.
- Related by the same group: Yang, Sun & Hardin (2011), Testing marginal homogeneity in
  clustered matched-pair data, *JSPI* **141**(3), 1313–1318, DOI
  [10.1016/j.jspi.2010.10.002](https://doi.org/10.1016/j.jspi.2010.10.002); (2012),
  Testing ratio of marginal probabilities…, *CSDA* **56**(6), 1829–1836, DOI
  [10.1016/j.csda.2011.10.025](https://doi.org/10.1016/j.csda.2011.10.025); (2012),
  Testing non-inferiority…in diagnostic medicine, *CSDA* **56**(5), 1301–1320, DOI
  [10.1016/j.csda.2011.06.019](https://doi.org/10.1016/j.csda.2011.06.019).

### 2.5 Yang, Sun & Hardin (2012) — **the CI paper. The closest thing to a Fagerland-equivalent.**

> Yang, Z., Sun, X. & Hardin, J. W. (2012). Confidence intervals for the difference of
> marginal probabilities in clustered matched-pair binary data. *Pharmaceutical Statistics*
> **11**(5), 386–393. DOI [10.1002/pst.1523](https://doi.org/10.1002/pst.1523).
> PMID 22684766. Cited 6x (per Semantic Scholar, 2026-08).

Abstract, verbatim opening (via Crossref JATS): "Although there are several available test
statistics to assess the difference of marginal probabilities in clustered matched-pair
binary data, associated confidence intervals (CIs) are not readily available."

- **What it does:** takes the existing family of clustered matched-pair *test* statistics
  and derives **Wald** and **Score** CIs from each; evaluates coverage by Monte Carlo.
- **Recommendation (abstract):** ICC-adjusted McNemar + its Wald or Score CI for **small to
  medium K**; that statistic becomes **conservative for large K**, where alternatives are
  preferred; in practice "a combination of the intracluster correlation coefficient-adjusted
  McNemar statistic with an alternative statistic is recommended."
- **Assumes:** whatever the underlying statistic assumes (Dirichlet-multinomial for the
  ICC-adjusted branch; assumption-free for the Obuchowski/Durkalski/Yang branches).
- **Closed form:** Wald yes; Score presumably closed-form or a simple root-find — **I could
  not verify this** (see §6).
- **Software:** none that I could find. `clust.bin.pair` does not implement CIs; `ratesci`
  does not implement clustered paired CIs; there is no Python implementation I located.
- **Systematic-comparison status:** this *is* the systematic comparison for the clustered
  case, but it is one small paper. There is **no** follow-up: of its 6 citations, four are
  applied papers and two are the authors' own kappa papers (Yang & Zhou 2014, *SiM* 33(15)
  2612–2633; Zhou & Yang 2014, *SiM* 33(14) 2425–2448). Nothing re-runs or extends the
  comparison.

### 2.6 Saeki & Tango (2011) — score CI for correlated proportions with **multiple raters**. Structurally the closest design to ours.

> Saeki, H. & Tango, T. (2011). Non-inferiority test and confidence interval for the
> difference in correlated proportions in diagnostic procedures based on multiple raters.
> *Statistics in Medicine* **30**(28), 3313–3327. DOI
> [10.1002/sim.4364](https://doi.org/10.1002/sim.4364). PMID 21953516.

- **Design:** each patient gets both procedures; **all images are read by all raters**.
  That is an n-patients x R-raters x 2-procedures binary array — **exactly the shape of our
  n-items x R-runs x 2-models array**.
- **What it gives:** a multinomial model for the matched-pair categorical data, from which
  they "derive a score-based full menu, that is, a non-inferiority test, **confidence
  interval** and sample size formula, for inference of the difference in correlated
  proportions." Monte Carlo shows the score test's size is closer to nominal than a Wald
  test and the score CI has better coverage than a Wald CI.
- **Why this matters to us:** Tango is an author. This is the nearest thing in print to
  "Tango's score interval, extended to repeated measurements per unit." If we want a
  score-interval lineage for the multi-run case, this is the citation.
- **Caveat on the design mapping:** their raters are *crossed* (rater j reads every
  patient, so there is a rater main effect); our runs are *nested/exchangeable* within
  item (run 3 of item 1 has no relationship to run 3 of item 2, unless we deliberately
  cross seeds). Their model may therefore carry a rater-effect term we do not want. **I
  could not verify the model equations** — paywalled.
- Follow-up: Saeki, Tango & Wang (2017). Statistical inference for noninferiority of
  difference in proportions of clustered matched-pair data from multiple raters. *J
  Biopharmaceutical Statistics* **27**(1), 70–83. DOI
  [10.1080/10543406.2016.1148709](https://doi.org/10.1080/10543406.2016.1148709). PMID
  26882055.
- **Software:** none found.

### 2.7 Rao & Scott (1992) — the canonical design-effect / effective-sample-size citation

> Rao, J. N. K. & Scott, A. J. (1992). A simple method for the analysis of clustered binary
> data. *Biometrics* **48**(2), 577–585. DOI
> [10.2307/2532311](https://doi.org/10.2307/2532311). PMID 1637980.

- "It is based on the concepts of **design effect and effective sample size** widely used in
  sample surveys, and **assumes no specific models for the intracluster correlations**"
  (abstract). Design effect estimated as the ratio of the variance of the *ratio estimate*
  of the probability to the standard binomial variance.
- **Scope:** *independent groups* of clustered binary data (homogeneity of proportions,
  dose-response, Mantel–Haenszel). **Not** the matched-pair case. So it is the right
  citation for "design effect / effective sample size for clustered binary data is a
  standard device", but **not** a citation for our paired multi-run interval.
- Kish, L. (1965). *Survey Sampling*. Wiley. — the original design-effect reference,
  if we want the `1 + (m-1)*rho` form attributed at source.

### 2.8 Design-effect CIs for a **single** clustered proportion (well developed, closed form, in R)

These are the marginal (one-arm) analogues. They matter because a MOVER/square-and-add
construction needs exactly these as inputs.

- Saha, K. K., Miller, D. & Wang, S. (2016). A comparison of some approximate confidence
  intervals for a single proportion for clustered binary outcome data. *International
  Journal of Biostatistics* **12**(2). DOI
  [10.1515/ijb-2015-0024](https://doi.org/10.1515/ijb-2015-0024). Compares profile
  likelihood, Wilson score, GEE (Zeger–Liang), and the **Rao–Scott ratio estimator**.
- Short, M. I., Cabral, H. J., Weinberg, J. M., LaValley, M. P. & Massaro, J. M. (2020). A
  novel confidence interval for a single proportion in the presence of clustered binary
  outcome data. *Statistical Methods in Medical Research* **29**(1), 111–121. DOI
  [10.1177/0962280218823231](https://doi.org/10.1177/0962280218823231). New **score-based**
  interval, better small-sample coverage. (See also the Zhang & Shan letter, *SMMR* 29(2)
  636–637, DOI [10.1177/0962280219840056](https://doi.org/10.1177/0962280219840056).)
- Shan, G. (2020). Accurate confidence intervals for proportion in studies with clustered
  binary outcome. *SMMR*. (Abstract not retrieved.)
- **Software:** `ratesci::clusterpci()` (CRAN, Pete Laud) — "asymptotic Score confidence
  intervals for a proportion estimated from a clustered sample", returning the ICC and a
  **variance inflation factor** (`xihat`); cites Saha et al. 2016 and Short et al. 2020.
  `ratesci` also has `scorepairci()` / `moverpairci()` for *unclustered* paired binomial
  data, but **nothing that combines paired + clustered**.

### 2.9 Design-effect CIs for a **two-independent-group** clustered difference

> Saha, K. K. & Wang, S. (2019). Confidence intervals for the difference in the success
> rates of two treatments in the analysis of correlated binary responses. *Biometrical
> Journal* **61**(4), 983–1002. DOI
> [10.1002/bimj.201700089](https://doi.org/10.1002/bimj.201700089). PMID 30843251.

- Proposes three interval procedures by "**direct extensions of recently proposed methods
  for independent binary data based on the concepts of design effect and effective sample
  size used in sample surveys**", each with four variance estimators, plus three
  complex-survey methods with different weighting schemes; extensive simulation.
- **This is the strongest published precedent for the *strategy* we used** — take a good
  unclustered interval and re-fit it with a design-effect-shrunken effective sample size —
  even though the contrast is unpaired rather than paired.

### 2.10 GEE and model-based routes

- Zeger & Liang GEE with an independence working correlation and a cluster-robust
  (sandwich) variance is the generic answer; for a risk **difference** this is an identity
  or linear-probability link, or a post-fit margin contrast. SAS documents both routes
  (SAS Usage Note 46997: `PROC FREQ COMMONRISKDIFF` stratification, or `PROC GEE`/`GENMOD`
  with the `Margins`/`NLMeans` macros).
- Known weakness, stated repeatedly in this literature: GEE sandwich intervals under-cover
  when the **number of clusters is small** — which for us is the small-n regime.
- Schwenke, C. & Busse, R. (2007). Analysis of differences in proportions from clustered
  data with multiple measurements in diagnostic studies. *Methods of Information in
  Medicine* **46**(5), 548–552. DOI [10.1160/me0433](https://doi.org/10.1160/me0433). A
  two-step (cluster-summary) approach covering within-patient, between-procedure and
  between-rater correlation; power-simulated against GEE and found "not inferior";
  explicitly aimed at "estimating proportions and differences in proportions for clustered
  data with multiple measurements … directly along with confidence intervals."
- Beta-binomial / Dirichlet-multinomial likelihood models exist as the parametric route
  (they are the assumed model behind Eliasziw–Donner's ICC estimator), but I found **no**
  paper that fits a beta-binomial specifically to get a paired-difference CI.

### 2.11 Other adjacent items found (lower relevance)

- Jin, H. & Lu, Y. (2009). Comparison of correlated proportions based on paired binary data
  from clustered samples. *JSPI* **139**(12), 4206–4212. DOI
  [10.1016/j.jspi.2009.06.005](https://doi.org/10.1016/j.jspi.2009.06.005). Abstract not
  retrievable (ScienceDirect 403, no abstract in Crossref/S2). Title is directly on-point;
  only ~4 citations.
- Shan, G. & Ma, C. (2014). Exact methods for testing the equality of proportions for
  binary clustered data from otolaryngologic studies. *Statistics in Biopharmaceutical
  Research* **6**(1), 115–122. DOI
  [10.1080/19466315.2013.861767](https://doi.org/10.1080/19466315.2013.861767).
- Shen, X. & Ma, C.-X. (2017). Testing homogeneity of difference of two proportions for
  **stratified** correlated paired binary data. *J Applied Statistics* **45**(8), 1410–1425.
  DOI [10.1080/02664763.2017.1371679](https://doi.org/10.1080/02664763.2017.1371679).
  (Stratified, not clustered.)
- Donner & Klar's cluster-randomised-trial literature is the general "cluster-level
  summary vs. individual-level analysis" framing; Donner (2007), The merits of breaking the
  matches, *SiM*, DOI [10.1002/sim.2662](https://doi.org/10.1002/sim.2662).
- ML-side: I found no closed-form clustered-paired-binary CI in the LLM-evaluation
  literature — everything there is cluster/item bootstrap. Closest adjacent item:
  Kotawala, "Resolution Diagnostics for Paired LLM Evaluation", arXiv 2605.30315 (May 2026)
  — power/sample-size diagnostics for paired LLM comparison with a clustering adjustment,
  not CIs.

---

## 3. Software summary

| package | clustered paired binary? | CIs? | notes |
|---|---|---|---|
| `clust.bin.pair` (CRAN 0.1.2, 2018, Gopstein) | yes — all 4 named tests | **no** | returns `htest` with statistic + p-value only; readable MIT source on GitHub, useful for cross-checking formulas |
| `ratesci` (CRAN, Laud) | no | yes for *either* paired *or* clustered, never both | `scorepairci`/`moverpairci` = unclustered paired; `clusterpci` = single clustered proportion with ICC + variance inflation factor |
| `contingencytables` (CRAN; companion to Fagerland et al.) | no | yes | mirrors the book, which covers unpaired/paired 2x2, rxc, ordered, paired cxc, **stratified** — I found no clustered chapter |
| SAS | partial | yes | `PROC FREQ COMMONRISKDIFF`; `PROC GEE`/`GENMOD` + `Margins`/`NLMeans` (Usage Note 46997) |
| Python | — | — | nothing found |

**There is no reference implementation of a clustered paired-binary CI in any language I
could find.** If we ship one it is genuinely new as software.

---

## 4. Is there a Fagerland-equivalent for the clustered case?

**No.** Concretely:

- Fagerland, Lydersen & Laake (2014), *SiM* 33(16) 2850–2875, is a large, heavily cited
  evaluation that produced a three-way recommendation the field now follows.
- The clustered analogue is Yang, Sun & Hardin (2012), *Pharm Stat* 11(5) 386–393 — 8
  pages, 6 citations, no methodological follow-up in 14 years, no software.
- The *testing* side is better served: Obuchowski (1998), Yang et al. (2010) and Wu (2019)
  each ran head-to-head Monte Carlo studies, and Yang et al. (2010) gives a clean
  cluster-size-conditional recommendation. But those are tests, not intervals.
- The `Fagerland` book (Chapman & Hall, 2017) covers stratified tables but, as far as I can
  determine from its published scope description, **not** clustered/repeated-measures
  paired binary.

So: for the multi-run paired binary CI there is a real, citable literature but **no settled
consensus**, and a defensible novelty claim if we want one.

---

## 5. Honest verdict on our design-effect approach

### 5.1 The strategy is standard and citable

"Take a good unclustered interval, replace n by an effective sample size
`n_eff = n / (1 + (m-1)*rho)`" is a recognised, published strategy with three levels of
support:

1. Kish (1965) — the design effect itself.
2. Rao & Scott (1992), *Biometrics* 48:577–585 — design effect + effective sample size for
   **clustered binary** data, model-free.
3. Eliasziw & Donner (1991) — the same device applied to **McNemar**, i.e. the paired
   binary case, with `C = 1 + (n_c - 1)*rho_tilde`; and Yang, Sun & Hardin (2012) — Wald
   and Score CIs built on exactly that ICC-adjusted McNemar statistic, **recommended for
   small-to-medium numbers of clusters**.
4. Saha & Wang (2019), *Biometrical J* 61:983–1002 — the same "extend an unclustered
   interval via design effect and effective sample size" recipe for a two-group difference.

So the *idea* is (a) standard and citable. We should cite Eliasziw–Donner and Yang et al.
(2012) as the paired-case precedent and Rao–Scott (and/or Kish) for the design-effect
device.

### 5.2 But our implementation is not the literature's design effect, and it is worth re-examining

Reading `evalstats/core/resampling.py::mj_floor_paired_ci_multirun_effective`
(lines ~2291–2380), the estimator is:

```
delta_i   = (b_i - c_i)/R                    # per-item mean paired difference
var_delta = Var(delta_i, ddof=1)             # BETWEEN-ITEM sample variance
u_i       = (b_i + c_i)/R ; within_i = u_i - delta_i^2 ; within_bar = mean(within_i)
rho       = clip(1 - within_bar/(var_delta*R), 0, 1)
R_eff     = R / (1 + (R-1)*rho)
between_latent = max(var_delta - within_bar/R_eff, 0)
total_var = between_latent/n + within_bar/(n*R_eff)
```

Two structural observations, both verified numerically against the installed code:

**(a) The base quantity is already cluster-level, so there is nothing left for a design
effect to correct.** `Var(delta_i, ddof=1)/n` **is** the centred cluster-robust variance of
`d_hat = mean_i delta_i`. Items are the independent sampling units; the within-item run
correlation is fully absorbed into the spread of the `delta_i`. This is precisely the
variance that Obuchowski / Durkalski / Yang estimate (they use an *uncentred* version,
valid only under H0; ours is centred, which is the correct form for a CI at `delta != 0`).
So the *unadjusted* estimator is already the right, citable, assumption-free thing.

**(b) Substituting the definitions, the `R_eff` step is either a no-op or a variance
inflation — it never shrinks.** With `rho_hat = 1 - within_bar/(var_delta*R)`:

- `within_bar / R_eff = var_delta * (1 - rho) * (1 + (R-1)*rho)`
- so `between_latent = var_delta * rho * [(R-1)*rho - (R-2)]`
- the `max(..., 0)` clamp fires iff `rho < (R-2)/(R-1)`.

Therefore:

| regime | resulting variance |
|---|---|
| `rho >= (R-2)/(R-1)` (clamp does not fire) | exactly `var_delta / n` — **the R_eff terms cancel algebraically** |
| `rho <  (R-2)/(R-1)` (clamp fires) | `var_delta * (1-rho)(1+(R-1)rho) / n` — **inflated**, factor `>= 1` |

The inflation factor `f(rho) = (1-rho)(1+(R-1)rho)` peaks at `rho* = (R-2)/(2(R-1))` with
value `1 + (R-2)^2/(4(R-1))`: 1.13x at R=3, 1.56x at R=5, 2.78x at R=10. It is 1 at
`rho = 0` and cancels for large `rho`.

Numerical checks (scratch scripts, `.venv` python, not committed):

- Closed form above reproduced the code's variance in 393/400 random cases (the 7 misses
  are the `1e-12` epsilon and the `rho` clamp boundaries).
- Over 400 random `(n in [10,80], R in [2,15])` draws from an independent-Bernoulli DGP,
  the method's variance was a **median 2.1x** (p90 3.3x, max 3.8x) the plain
  `Var(delta_i)/n` — i.e. a **median 1.46x SE inflation**.
- Realised *interval width* inflation is much smaller because the `z^2 * s_hat / n^2`
  discordance-floor term dominates when discordance is low: in a 1500-rep null-coverage
  check under a latent-normal item-effect DGP, widths were only 4–21% larger, and **both**
  variants over-covered (0.96–0.999 at nominal 0.95).

**Caveat on those numbers:** these are quick scratch simulations at the null with two
ad-hoc DGPs, not the project harness. They are enough to establish the algebraic claim in
(b) and to show the direction of the effect; they are **not** a calibration verdict. The
harness (`simulations/harness/cases/ci_paired.py`) is the place to settle whether the
inflation is buying anything.

### 5.3 Classification

Against the three options in the brief:

- **(a) standard and citable?** The *design-effect strategy* — yes. *Our particular
  formula* — no; it does not appear anywhere in this literature, and it is not
  Eliasziw–Donner's `1 + (n_c - 1)*rho_tilde` (different ICC, different target quantity,
  applied to a different base variance).
- **(b) a reasonable approximation?** Yes, in the sense that it is conservative — it never
  under-states the cluster-level variance, and coverage in my quick checks was at or above
  nominal. It is not wrong in a coverage-damaging direction.
- **(c) naive?** In one specific respect, yes: the design effect is applied on top of a
  variance that has already accounted for the clustering, so it double-counts. The
  double-count is masked by the `max(..., 0)` clamp turning into an inflation rather than a
  contradiction. A reviewer who works through the algebra will notice that the `R_eff`
  machinery cancels in one branch and inflates in the other, and will ask why.

---

## 6. What I could NOT verify

Stated plainly, because a wrong citation is worse than no citation:

1. **The actual CI formulas in Yang, Sun & Hardin (2012).** Paywalled at Wiley; no open
   copy on Europe PMC, IA Scholar, or arXiv. I have the full abstract (verbatim, from the
   Crossref JATS record) and the recommendation, but **not** the Wald/Score constructions,
   their ICC estimator, or the simulation grid (K, cluster size, ICC ranges). Do not cite
   any formula-level claim about this paper without reading the PDF.
2. **The Saeki & Tango (2011) model and score CI.** Paywalled. I have the abstract only. In
   particular I could not confirm whether their rater effect is crossed (which would make
   the model a poor fit for exchangeable runs) or how their score interval relates to
   Tango (1998).
3. **Obuchowski (1998) and Durkalski (2003) in the original.** I read their statistics from
   the `clust.bin.pair` R source (MIT, readable) and their abstracts from Europe PMC, not
   from the papers. The R source is a third-party reimplementation; treat the formulas as
   "as implemented in `clust.bin.pair`", not "as published", until checked against the PDFs.
4. **Eliasziw & Donner's ICC estimator details** — same caveat; read from `R/eliasziw.R`.
5. **Jin & Lu (2009), *JSPI* 139:4206–4212** — could not retrieve the abstract at all
   (ScienceDirect returned 403; Crossref and Semantic Scholar have no abstract). Title is
   directly relevant. Someone with library access should check it.
6. **Whether the Fagerland/Lydersen/Laake book has a clustered section.** Publisher and
   contingencytables.com both returned 403. Judged "no" from the published scope blurb
   (which lists unpaired/paired 2x2, rxc, ordered, paired cxc, stratified) — not confirmed
   from a TOC.
7. **Blocked sites, noted rather than worked around:** `pubmed.ncbi.nlm.nih.gov` (cookie
   wall), `pmc.ncbi.nlm.nih.gov` (reCAPTCHA — not bypassed), `sciencedirect.com` (403),
   `onlinelibrary.wiley.com` (403), `routledge.com` (403), `contingencytables.com` (403).
   Metadata and abstracts above came from the Crossref and Europe PMC REST APIs and from
   CRAN/GitHub, all of which served content normally.

---

## 7. Recommendations

**Citations to add regardless of what we implement.** The paper currently has a gap here;
these four sentences' worth of prior work should be acknowledged:

- Eliasziw & Donner (1991) — the ICC-adjusted McNemar; the original design-effect
  correction for paired binary clustering.
- Obuchowski (1998) and Durkalski et al. (2003) — assumption-free cluster-level statistics.
- Yang, Sun & Hardin (2010) — modified Obuchowski; **explicitly recommended for equal
  cluster sizes**, which is our design.
- Yang, Sun & Hardin (2012) — the only CI paper; its ICC-adjusted-McNemar recommendation is
  the precedent for a design-effect interval.
- Rao & Scott (1992) (and Kish 1965) — for the design-effect / effective-sample-size device.
- Saha & Wang (2019) — precedent for "extend an unclustered interval by design effect".
- Saeki & Tango (2011) — if we frame our method as a score interval in Tango's lineage
  extended to repeated measures, this is the paper a reviewer will expect to see.

**On the method itself, in priority order:**

1. **Re-run `ci_paired` with the `R_eff` step removed** (variance = `Var(delta_i)/n` plus
   the existing score shrinkage and discordance floor) and compare coverage/width/power
   against the current `mj_floor_er`. If the plain cluster-level version holds coverage,
   drop `R_eff`: it is a term we cannot cite, it double-counts, and removing it buys
   width. If it *loses* coverage, we now know exactly what the inflation is paying for and
   can say so.
2. **Add Yang et al.'s (2010) modified-Obuchowski statistic as a harness comparator**, and
   the Wald CI derived from the *centred* cluster-level variance. It is four lines of code
   (formula in §2.4), it is the literature's recommended statistic for equal cluster sizes,
   and having it in the comparison table is exactly the kind of thing a Statistics in
   Medicine-literate reviewer will look for.
3. **Frame the contribution honestly**: the clustered matched-pair *testing* problem is
   solved; the *interval* problem has one small 2012 paper, no follow-up, and no software
   in any language. A well-calibrated closed-form interval for R repeated runs, with a
   proper simulation study, is a real gap — but only if we position it against Yang et al.
   (2012) and Saeki & Tango (2011) rather than against Fagerland et al. (2014) alone.
4. **Get the PDFs** of Yang et al. (2012), Saeki & Tango (2011), Obuchowski (1998) and
   Eliasziw & Donner (1991) before any formula-level claim goes into the paper. Items 1–5
   in §6 are the specific things to check.
