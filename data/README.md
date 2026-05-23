# Data

This folder is where the raw Kaggle CSV lands after download.

## Automatic download

Running `python main.py` from the project root will try to pull the dataset using the Kaggle API. For that to work, you need a Kaggle account and an API token saved at:

* Linux or macOS: `~/.kaggle/kaggle.json`
* Windows: `C:\Users\<your-user>\.kaggle\kaggle.json`

You can generate the token from your Kaggle account settings page.

## Manual download fallback

If you would rather skip the API setup, grab the file by hand from:

> https://www.kaggle.com/datasets/shubh0799/churn-modelling

Then drop `Churn_Modelling.csv` into `data/raw/`. The pipeline will pick it up automatically and skip the API call.

## Schema (quick reference)

| Column            | Description                                              |
|-------------------|----------------------------------------------------------|
| RowNumber         | Sequential row id, useless for modeling                  |
| CustomerId        | Bank-internal customer identifier                        |
| Surname           | Customer surname (dropped before modeling)               |
| CreditScore       | 350 to 850, standard credit bureau range                 |
| Geography         | France, Germany, or Spain                                |
| Gender            | Male / Female                                            |
| Age               | Years                                                    |
| Tenure            | Years as a customer                                      |
| Balance           | Current account balance in euros                         |
| NumOfProducts     | Number of bank products held (1 to 4)                    |
| HasCrCard         | 1 if the customer holds a credit card                    |
| IsActiveMember    | Bank's internal activity flag                            |
| EstimatedSalary   | Annual salary estimate in euros                          |
| Exited            | Target: 1 if the customer churned, else 0                |
