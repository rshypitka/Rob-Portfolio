# ExtraaLearn Companion Guide

A walkthrough of the submission notebook, section by section. For each part it covers what the
code accomplishes, how the approach works at a concept level, and the decisions behind it with the
reasoning. Read this alongside the notebook and you should come away knowing what was done, what was
chosen, and why.

This is not a line-by-line code narration. Obvious mechanics are skipped. The focus is judgment.

---

## 1. Business Problem Overview

**What it does.** Frames the problem before any code: ExtraaLearn generates more leads than sales can
work, every lead sits in one undifferentiated queue, and the cost lives in the human follow-up (a rep
calling or emailing). The objective is stated as three things: build a model to flag likely converters,
find the drivers of conversion, and describe what a likely converter looks like.

**Decisions and defense.**
- The objective is aligned to the three goals in the official brief, not paraphrased. This matters
  because the grade rewards answering the actual question asked.
- The recall-first framing is established here, before modeling: missing a likely converter costs more
  than a wasted call. That single business judgment drives every modeling choice downstream (the scoring
  metric, the model selection). Stating it up front makes the later choices defensible rather than arbitrary.

---

## 2. Setup & Library Imports

**What it does.** Imports the libraries, sets display and plotting defaults, and prints versions.

**Decisions and defense.**
- Imports are grouped by function (core, viz, model selection, models, metrics) for readability.
- Only libraries actually used are imported. Unused imports were removed during the audit, which keeps
  the code honest about what the notebook depends on.
- `warnings.filterwarnings("ignore")` suppresses non-critical library warnings so the output stays
  focused on results. This is convention for a submission notebook; the template itself does the same.
  Worth knowing it suppresses rather than resolves warnings, so the underlying run was also checked to
  be genuinely clean, not just quieted.

---

## 3. Data Understanding & Audit

**What it does.** Reads the CSV and runs a first-pass audit: shape, dtypes, null check, duplicate check,
and split summary statistics for numeric vs categorical features.

**How it works.** `describe()` is split into numeric and categorical views rather than one combined table,
because a single combined describe is hard to read across mixed types.

**Decisions and defense.**
- The read path is relative (`pd.read_csv("ExtraaLearn.csv")`), so the notebook runs on any machine with
  the data file beside it. This was deliberate for portability between machines.
- `ID` is identified as a row identifier and set aside: kept for potential lead tracking but excluded from
  model training, since an identifier has no predictive value and would only add noise.
- The audit observation flags the mean-vs-median gap in the engagement features (e.g. time on website mean
  724 vs median 376) as a right-tail signal. This is an honest read of what the summary table shows, not a
  preprocessing decision; the decision comes later.

---

## 4. Exploratory Data Analysis

**What it does.** Two phases. Univariate (target balance, numeric distributions, categorical counts) then
multivariate (correlation, conversion rate by category, numeric features split by conversion status).

**How it works.**
- Numeric distributions use a paired boxplot-over-histogram layout so spread and outliers read in one view.
- Conversion-rate-by-category computes the mean of the binary target within each category, which is exactly
  the conversion rate for that group.
- The correlation analysis sits under multivariate because it relates features to the target, which is
  bivariate work, not univariate.

**Decisions and defense.**
- The target is ~30% positive (imbalanced). This is why accuracy is rejected as the headline metric: a
  model predicting "no one converts" would score ~70% accurate while being useless. Precision, recall, F1,
  ROC-AUC, and PR-AUC carry the evaluation instead. The imbalance is judged not severe enough to need
  resampling like SMOTE; 70/30 is workable directly.
- The strongest categorical signals surface here and hold up later: website-first leads convert at ~45.6%
  vs ~10.5% for mobile-app-first, high profile completion at ~41.8% vs ~7.5% for low. These same features
  top the model importance later, which is the consistency that makes the findings credible.
- The correlation observation makes a point worth keeping: low linear correlation does not mean a feature
  is unimportant, because correlation only captures straight-line relationships. This pre-justifies why
  tree models (which capture non-linear splits) find signal the heatmap understates.

---

## 5. Preprocessing & Feature Engineering

**What it does.** Confirms no missing values, reviews outliers (IQR), checks skewness, engineers features,
splits the data, then applies data-driven transforms to the training set only.

**How it works.** The defining structure is the split-first, learn-from-train-only workflow:
- Rule-based steps (binary encoding of Yes/No fields, the marketing touch count) run BEFORE the split,
  because they leak nothing: they map fixed values, they don't learn anything from the data.
- Data-driven steps (outlier caps via training IQR, the MinMax engagement scaler, the high-intent time
  threshold) run AFTER the split, learned from training data only, then applied to test.
- The final ColumnTransformer scales numeric features and one-hot encodes categoricals, fit on train only.

**Decisions and defense.**
- This split-first ordering is the most important decision in the section, and arguably the notebook. If
  outlier bounds, the scaler, or the median threshold were learned from the full dataset, test information
  would bleed into preprocessing and the later model scores would be optimistic. Learning everything from
  train only keeps the test set genuinely unseen, so the reported scores reflect real performance.
- Outliers are capped, not removed. A lead with very high website visits is plausibly a real, highly
  engaged person, not a data error, so deleting them would throw away valid signal. Capping limits their
  leverage on the model without discarding the rows.
- Skewness is noted but no log transform is applied, since tree-based models are insensitive to monotonic
  feature scaling. The capping handles the heaviest tails, which is enough for trees.
- Three features are engineered (marketing_touch_count, engagement_score, high_intent_flag). They are built
  on a deliberate principle: rule-based ones before the split, data-driven ones after. Whether they help is
  left for the models to decide rather than assumed. (They turn out near-zero in importance; see below.)
- The IQR lower bounds compute as negative for some features (e.g. website visits), which is just the
  formula. Since no values fall below zero, those bounds never bind. Harmless arithmetic, not a data issue.

---

## 6. Decision Tree Model (required model 1)

**What it does.** Builds a baseline tree, tunes it with GridSearchCV scored on recall, then reads feature
importance.

**How it works.**
- A shared `evaluate_model` helper fits a pipeline and returns the six metrics, so every model is scored
  the same way. A shared StratifiedKFold gives consistent cross-validation folds across all tuned models.
- The baseline uses `class_weight="balanced"` to handle the 70/30 imbalance inside the estimator.
- Tuning searches depth, leaf size, split size, and criterion, scoring on recall so the search optimizes
  for catching converters.

**Decisions and defense.**
- Scoring the tuning on recall (not accuracy or roc_auc) is the direct expression of the recall-first
  business framing. The brief says missing a converter is the costly error, so the model is optimized to
  minimize that. This was sanity-checked: a pure-recall objective can degenerate into flagging everyone,
  so the tuned result was inspected to confirm precision held (it did, ~0.60) and ranking improved.
- Baseline vs tuned tells a clean improvement story: recall rose from 0.69 to 0.90 and PR-AUC from a weak
  0.55 to 0.68, while precision gave up only a few points. That is the recall-first tradeoff working as
  intended, not a model breaking.
- Feature importance is mapped back to readable names through the fitted ColumnTransformer
  (`get_feature_names_out`), because raw importances align to the 23 post-encoding columns, not the
  original 14. The chart shows the top features the tree actually used (it splits on only six).
- The engineered features land at zero importance in the tree. This is stated plainly rather than glossed:
  the tree found nothing in them it could not already get from the raw features they were built from. They
  stay in the notebook as documented attempts. Stating this honestly is more credible than quietly keeping
  weak features.
- The section closes with a plot of the tuned tree itself (`plot_tree`). At depth 3 it is small enough to
  read every split, so it shows the actual decision rules, not just which features matter. The root splits
  on the first-interaction channel, then both branches split on time on site, refining on profile completion
  and activity. One leaf isolates 715 not-converted leads at zero entropy (perfectly pure). This is the one
  model whose structure is genuinely visualizable; a forest or boosted sequence is not, which is why only the
  tree gets a structure plot. It makes the interpretability argument concrete rather than asserted.

---

## 7. Random Forest Model (required model 2)

**What it does.** Same arc as the tree: baseline, GridSearchCV tuned on recall, feature importance.

**How it works.** A forest builds many trees on random subsets of rows and features, then averages them.
That averaging usually generalizes better than a single tree and resists overfitting. Tuning searches the
forest-specific knobs: number of trees, depth, leaf size, and feature sampling per split.

**Decisions and defense.**
- The grid uses RF-appropriate hyperparameters (n_estimators, max_features) rather than reusing the tree's
  grid. The number of trees and the per-split feature sampling are what make a forest a forest; tuning a
  forest as if it were a single tree would miss the point.
- Same recall scoring, same CV, same helper as the tree, so the two required models are directly comparable.
- The tuned forest reaches recall 0.87 with the cleanest ranking of any model (PR-AUC 0.83) and stays
  reasonably precise (0.68). It trades a little recall for much better balance than the tree.
- Importance agrees with the tree on the top three drivers: time on website, first interaction, profile
  completion. Two independently built and tuned models landing on the same drivers is strong evidence the
  signal is real and not a quirk of one model.
- The forest spreads tiny importance across many features (normal for an ensemble), so the chart shows the
  top contributors rather than every non-zero bar. Both importance charts use the same top-N rule for
  consistency. The engineered features again sit near zero, consistent with the tree's verdict.

---

## 8. Additional Analysis: Boosted Models (going further)

**What it does.** Tunes Gradient Boosting and XGBoost, then compares all four tuned models in one table.

**How it works.** Boosting builds trees in sequence, each correcting the last, which can extract more signal
than a forest's independent-and-average approach. Both are tuned with GridSearchCV on recall to keep the
comparison fair.

**Decisions and defense.**
- This section exists to answer one honest question: is the random forest near the performance ceiling for
  this data, or is it leaving something on the table? Boosting is the natural test. Framing it as a question
  with both answers being informative (a real win would mean go complex; a tie means the simple model is
  fine) is more rigorous than just trying more models for show.
- Tuned models only here, no baselines. The required models earned the baseline-then-tuned treatment because
  building and improving them is the graded task. Here the question is narrower (does boosting beat the
  forest), so only the tuned boosters are needed. Adding baselines would be bulk without purpose.
- Gradient Boosting is evaluated directly on the grid's fitted estimator, NOT through the shared helper.
  This is the one deliberate deviation in the notebook, and it is documented in the code. GB handles the
  class imbalance through fit-time sample weights, and the helper's refit would discard those weights,
  producing a misleadingly low recall. Evaluating the already-fitted model keeps the tuned, balanced version
  intact. XGBoost is evaluated the same way for consistency, though its imbalance handling
  (scale_pos_weight) is a hyperparameter that would survive a refit anyway.
- The finding: all four models cluster tightly. Boosting does not beat the forest by any margin worth the
  added complexity. The conclusion is that the simpler, interpretable tree-based model is the right one to
  ship. This is explicitly tied to the current data: a larger dataset with richer features could shift the
  answer, which is where a boosted model tends to pull ahead. That keeps the conclusion honest and bounded
  rather than a blanket verdict.

---

## 9. Recommendations

**What it does.** Four business recommendations: prioritize outreach by model score, treat website
engagement as an intent signal, rework the mobile-app-first path, and drive profile completion earlier.

**Decisions and defense.**
- The recommendations are anchored to the model evidence. Three of the four map directly to the top three
  drivers both models agreed on; the fourth (prioritize by score) is the operational answer to the project's
  stated goal of allocating resources efficiently.
- The driver claims are supported with the EDA conversion rates (e.g. website-first ~45% vs mobile ~10%),
  which give direction. Feature importance shows a feature matters; the EDA shows which way it cuts. Using
  both keeps the recommendations grounded in evidence rather than assertion.
- Four focused recommendations rather than a longer list. Each is specific and actionable, which is what the
  business-recommendation criterion rewards. Padding with weaker recommendations would dilute the strong ones.

---

## 10. Limitations

**What it does.** Four honest constraints: single-snapshot data, importance is not causation, weak channel
signal, and the recall-first precision tradeoff.

**Decisions and defense.**
- The causation point is the analyst's discipline showing: the model finds mobile-app-first leads convert
  less, but that does not prove the app causes lower conversion. It could be audience mix or intent. Naming
  this prevents over-reading the model.
- The recall tradeoff is stated plainly: tuning for recall accepts more false positives, so precision sits
  in the low-to-mid 60s and a meaningful share of flagged leads will not convert. Owning the cost of the
  chosen tradeoff is more credible than presenting the model as flawless.

---

## 11. Conclusion & Executive Summary

**What it does.** Recaps the project: the random forest is the model to carry forward, the drivers are
consistent across both required models, boosting did not improve on them, and a likely converter is
profiled (spends real time on site, came through the website, completed their profile).

**Decisions and defense.**
- The random forest is named as the pick, consistent with the Additional Analysis conclusion. The notebook
  does not crown a boosted model it earlier showed was no better; internal consistency holds.
- The converter profile explicitly closes the loop on the third stated objective, so all three goals from
  the overview are visibly delivered.

---

## The through-line

A few decisions run the length of the notebook and tie it together:

1. **Recall-first**, established in the overview, drives the scoring metric and the model evaluation.
2. **Leakage-safe preprocessing** (split first, learn from train only) makes every reported score trustworthy.
3. **Honesty about what did not work** (engineered features near zero, boosting no better than the forest)
   is stated plainly rather than buried, which makes the claims that do hold more believable.
4. **Interpretability favored over marginal complexity**: the forest is chosen over boosting because the
   performance is equivalent and the simpler model is easier to explain and trust.
