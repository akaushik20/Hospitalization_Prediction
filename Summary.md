### What I found in the data and what I did about it

I started with EDA (`eda.py`) to explore the data. I used ydata-profiling, which is a great library for generating an EDA report on smaller datasets ([C:\Users\arpit\OneDrive\Desktop\My Files\GitHub\Hospitalization_Prediction\output\eda\patients_profile.html](C:\Users\arpit\OneDrive\Desktop\My Files\GitHub\Hospitalization_Prediction\output\eda\patients_profile.html)). My observations:

- The dataset has 20 features and 1 binary target variable.
- The dataset is imbalanced: the positive class is only 8% and the negative class is 92%.
- There are 89 duplicate rows (0.2%).
- The data has both categorical and numeric features.
- `egfr` and `hba1c` have a high percentage of missing values.
- The following features are highly correlated:
  - `discharge_summary_count` is highly correlated with the target, `hospitalized_next_year`
  - `n_active_medications` is highly correlated with `n_chronic_conditions`
  - `n_chronic_conditions` is highly correlated with `age` and 2 other fields
  - `n_encounters_last_year` is highly correlated with `n_chronic_conditions`

**Steps I took:**

- Dropped the 89 duplicate rows, since it's a small number and won't impact overall performance.
- Removed `discharge_summary_count` because it's a leaky feature. When I ran the model with it, the F1 score was 0.95.
- If I had more time, I would have removed one feature from each highly correlated pair to reduce the risk of multicollinearity.

### Modeling approach and why

- I used XGBoost because it performs well on imbalanced classification problems.
- I set XGBoost's `scale_pos_weight` parameter to the ratio of negative to positive classes in the training set, so the model gives more weight to the minority class. I don't think oversampling or undersampling actually helps much in an imbalanced classification problem.
- Instead of encoding the categorical columns with a label encoder or target encoder, I converted them to the pandas `category` type and set `enable_categorical=True`. This lets XGBoost split on groups of categories directly, without imposing an artificial order (as label encoding does). If I had more time I would have experiented more with target_encoding 

### How I evaluated it

- I evaluated the model on a held-out test set using ROC-AUC and F1 score.
- ROC-AUC measures how well the model ranks positive patients above negative ones, and F1 balances precision and recall on the minority class, which matters because only 8% of patients are positive.

### How you'd explain what this model does, and how good it is, to a customer's care team (non-ML audience)

Think of this situation, when we do not have any data science model and we are taking a andom guess on which patient will be hospitalized. In that case we have a 8% chance of identifying a hospitalization correctly. 
Now lets assume a base min model which gives us an F1 score of 0.4, that is 200% better than random guess. So a naive model can provide you a 200% better chance to predict the right outcome 