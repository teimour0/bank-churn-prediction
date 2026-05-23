# ChurnRadar: Retail Banking Customer Intelligence

A typical retail bank loses roughly one in five customers every year. Most of them leave quietly. By the time the relationship manager notices, the money is already sitting in someone else's account down the street.

This project came out of conversations I kept having with friends who work in retail banking. The same question came up over and over: which customers are about to walk, and what is that actually costing us in dollar terms?

So I built ChurnRadar. It pairs predictive modeling with customer segmentation, and adds a rough but honest estimate of how much revenue is at risk. The whole pipeline runs end to end from a Kaggle dataset, and there is a small Streamlit dashboard at the end so a non-technical reader can poke around.

## What the project actually does

1. Pulls the Churn Modelling dataset from Kaggle (10,000 customers from a European bank across three countries).
2. Cleans the raw file and adds a handful of engineered features that turned out to be useful while exploring the data. Things like balance-to-salary ratio, age buckets, a "sticky customer" flag, and a flag for customers carrying real money but classified as inactive.
3. Runs a short exploratory pass that saves every chart into `reports/figures/`, so they can be reused later in slides or a writeup.
4. Groups customers into behavioural segments using KMeans. A single churn probability does not tell you much about *who* is leaving; segments give you a vocabulary to work with.
5. Trains three models (logistic regression, random forest, gradient boosting), compares them honestly, and keeps the best one.
6. Translates the model output into something a business audience cares about: estimated revenue at risk, broken down by segment.
7. Serves the whole thing through a Streamlit dashboard with a modern dark theme.

## Why this dataset

The Kaggle Churn Modelling file is small, clean, and almost too tidy. That is fine for a portfolio project, because the goal is to show analytical thinking and not to spend three weeks fixing CSV encoding bugs. The dataset has a believable mix of numerical and categorical features, and the target is balanced enough to make modeling interesting (about 20% churn) without being trivially easy.

If you want a harder version of the same problem, swap the loader for the *Bank Customer Churn Prediction* dataset by Radheshyam Kollipara. That one has more features and the rest of the pipeline will still run.

## Project layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # one-shot pipeline runner
├── dashboard.py             # Streamlit app
├── .streamlit/
│   └── config.toml          # theme settings
├── data/
│   ├── README.md            # how to get the CSV
│   └── raw/                 # placed here after Kaggle download
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── eda.py
│   ├── segmentation.py
│   ├── modeling.py
│   └── economic_impact.py
└── reports/
    ├── findings.md          # written narrative of what I found
    └── figures/             # auto-generated charts
```

## Getting it running

You will need Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Grab your Kaggle API token from `kaggle.com/settings` and drop `kaggle.json` into `~/.kaggle/`. On Windows that is `C:\Users\<you>\.kaggle\`. Then:

```bash
python main.py
```

That single command downloads the data, runs the full pipeline, writes figures and a model artifact, and prints the headline numbers. After it finishes:

```bash
streamlit run dashboard.py
```

If you would rather not deal with the Kaggle API, the script falls back to looking for `data/raw/Churn_Modelling.csv` and will happily use a file you placed there manually.

## Headline findings

I am repeating these here so a recruiter scanning the README does not have to dig through the report to see what came out of the analysis.

* Customers between 45 and 65 churn at almost twice the rate of customers under 35, even though they hold larger balances on average. That makes them the single most expensive segment to lose.
* Holding three or four products is actually a *worse* signal than holding one or two. Counter-intuitive, but the data is clear, and it lines up with what I have read about cross-sell saturation in retail banking.
* German customers churn at 32%, which is roughly double the rate for France and Spain. They also carry higher balances. Worth investigating as a country-specific issue rather than a modeling artifact.
* Gradient boosting wins on ROC-AUC (around 0.86 in my runs), but logistic regression is within a couple of points and is much easier to explain in a meeting. The dashboard lets you pick which one to use.
* Estimated annual revenue at risk for the high-probability segment sits around \$3.3M on this simulated portfolio. The number matters less than the framing. Putting a dollar sign on a model output changes how stakeholders react to it.

A longer writeup of all this is in [reports/findings.md](reports/findings.md).

## Things I would do differently with more time

* The economic impact model assumes a flat customer lifetime value of \$2,000. A proper version would condition CLV on product mix and tenure.
* I did not run a serious hyperparameter search beyond a small grid. With more compute I would do a proper Bayesian search and confirm whether the AUC gap between gradient boosting and logistic regression is real or just noise.
* The segmentation uses KMeans because it is simple to explain. For a real project I would compare it against Gaussian mixture and a hierarchical method, then pick based on silhouette plus business interpretability.
* The dashboard re-trains the model on launch right now. Caching the model artifact properly is the obvious next step.

## A note on the data

The Kaggle dataset is anonymised and feels synthetic in places. Treat the dollar figures as illustrative, not as a forecast for any real bank. The point of the project is the method, not the magnitude.
