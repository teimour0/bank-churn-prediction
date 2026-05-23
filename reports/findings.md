# Findings: ChurnRadar

This is the long-form version of what I learned while building the project. I am writing it as I would write a short internal memo, not a slide deck, so the tone is plain and the structure follows the analytical work rather than a template.

## The shape of the dataset

Ten thousand customers across France, Germany, and Spain. About 20% of them churned. That is high enough to be a real business problem but not so high that the dataset is degenerate. The columns cover the usual things: demographics, account behaviour, and a few bank-internal flags like "active member" whose precise definition I could not find documented anywhere.

A couple of things stood out before I touched any model.

Balance is bimodal. A meaningful share of customers carry a zero balance, and the rest cluster around the 100,000 to 130,000 euro mark. Treating balance as a single continuous variable hides that, which I think is why simple linear baselines underperform on this dataset.

The "active member" flag is independent of balance. You can have 200,000 euros sitting in an account and still be classified as inactive. That felt wrong to me, and the data agrees. Customers in that overlap churn at a higher rate than either group on its own. That observation became the `DormantWithMoney` engineered feature.

## What predicts churn here

The three usual suspects show up clearly.

**Age.** Customers between 45 and 65 churn at almost twice the rate of customers under 35. This is the single sharpest signal in the data and it is not subtle. The age-group bar chart in `reports/figures/03_churn_by_age_group.png` makes the gap obvious without any modeling required. The natural interpretation is that mid-life customers shop their financial relationships more aggressively than younger ones, who tend to stay where their salary lands by default. That is speculation on my part, but the pattern is real.

**Geography.** German customers churn at 32%, compared to about 16% for French and Spanish customers. Their balances are higher on average, which makes them disproportionately expensive to lose. Without more context I cannot say whether this is a product issue, a service issue, or a pricing issue specific to Germany. It is the obvious thing to investigate first if this were a real engagement.

**Product count.** This is the counter-intuitive one. Customers holding three or four products churn far more than customers holding one or two. Common cross-sell intuition would predict the opposite. My best guess is that customers with three or four products bought them reluctantly, or were sold them under conditions that did not match their needs, and the dataset is picking up the aftermath of that. Either way it is a useful flag for the model and a useful question for the business.

Tenure, credit score, and salary all carry signal, but much weaker than the three above. Gender barely moves the needle.

## How the models compared

I trained three baselines with stratified train/test splits.

| Model               | ROC-AUC | Precision | Recall | F1   |
|---------------------|---------|-----------|--------|------|
| Gradient boosting   | ~0.86   | 0.78      | 0.49   | 0.60 |
| Random forest       | ~0.85   | 0.75      | 0.51   | 0.61 |
| Logistic regression | ~0.77   | 0.39      | 0.69   | 0.50 |

(Exact numbers move slightly between runs because I did not pin a global seed everywhere.)

A few notes on this table that matter more than the numbers themselves.

Logistic regression has higher recall and lower precision than the tree-based models. That is the class-weighting at work. The linear model is reaching further to catch churners and accepting more false alarms. Depending on whether the retention team has the capacity to call false positives, that trade is sometimes the right one to make. The dashboard exposes the threshold so a user can see what happens at different cutoffs.

Gradient boosting wins on AUC by a small margin. In a real meeting I would probably argue for logistic regression anyway. The coefficients are directly interpretable and the lift in AUC is not large enough to justify the loss in explainability. The point of this project is to show that I know how to run the comparison and present the trade-off, so the dashboard keeps the gradient booster as the default.

I did not do serious hyperparameter tuning. With a dataset this size the marginal AUC gain from a Bayesian search would likely be small and would not change the story.

## Putting a dollar on the model

The economic impact module multiplies each customer's churn probability by a flat customer lifetime value of 2,000 dollars and aggregates by segment. With the random forest running on the full portfolio, total expected loss lands around 5.9 million dollars, with roughly 3.3 million of it concentrated in the high-probability band (probability greater than or equal to 0.5, about 2,200 customers).

The CLV assumption is the obvious thing to attack. A real version of this would condition CLV on product mix, tenure, and probably geography too. What the framing does well even in this rough form is force a conversation about *which* customers to chase. The high-balance, mid-age, inactive cluster is small in headcount but disproportionate in dollar terms. A retention budget that ignores that is leaving money on the table.

## Segments

KMeans with k=4 lands on four clusters that I can describe without resorting to a centroid table.

1. **Mid-life passive saver.** High balance, classified as inactive. Around 3,200 customers and a 28.6% churn rate. The cluster that deserves the bulk of retention spend in dollar terms.
2. **Mid-life engaged saver.** Same balance profile but active. About 2,900 customers and a much healthier 13% churn rate. The "stable" core.
3. **Mid-life transactor.** Essentially zero balance, multi-product, half-active. About 2,800 customers and the lowest churn rate in the portfolio at around 12%. Uses the bank for flow rather than savings.
4. **Senior engaged saver.** Older customers, mid-balance, active. The smallest group at around 1,100 customers but the highest churn rate at 36.3%. Disproportionately important because of the age and balance combination.

The exact cluster IDs change run to run because I did not pin every seed. The shape of the segmentation is stable.

## What I would do next

In rough priority order:

* Recompute CLV per customer rather than using a flat assumption. The headline dollar figure changes meaningfully when you do.
* Investigate the German churn rate as a specific business question. Pricing? Branch network? Competitor activity? The dataset alone cannot answer that, but it can sharpen the question for whoever can.
* Run a proper hyperparameter search on the gradient booster with stratified cross-validation, mostly to confirm that the AUC gap to logistic regression is real and not noise.
* Add SHAP explanations to the dashboard so a single-customer score comes with a "why" alongside the number. I prototyped this and dropped it for scope reasons, but it is the obvious next feature.

That is the project. The code is small enough to read in an afternoon, the analysis lines up with what the business would actually want to see, and the dashboard turns it into something a non-technical reader can poke at without having to read any Python.
