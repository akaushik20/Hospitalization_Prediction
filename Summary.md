### What I found in the data and what I did about it

I started with EDA (`eda.py`) to explore the data. I used ydata-profiling, which is a great library for generating an EDA report on smaller datasets I used ydata-profiling to generate an EDA report ([EDA report](output\eda\patients_profile.html)). My observations:

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
- Selected the top 10 features by gain, since they perform as well as the full feature set (5-fold CV on the training set):
  - F1 score with all features: 0.7676 ± 0.0048
  - F1 score with top 10 features: 0.7720 ± 0.0081

  When a smaller set of features performs just as well, adding features that bring little or no improvement isn't good practice. It makes the model more complicated and adds more to monitor in production.

**Model Results**

- **F1-Score:** 0.3080
- **Average Precision Score:** 0.3585
- **ROC-AUC:** 0.7342

### How I evaluated it

- I evaluated the model on a held-out test set using ROC-AUC and F1 score.
- ROC-AUC measures how well the model ranks positive patients above negative ones, and F1 balances precision and recall on the minority class, which matters because only 8% of patients are positive.

### How you'd explain what this model does, and how good it is, to a customer's care team (non-ML audience)

- **What this model does:** The model looks at each patient history, for example recent visits, existing conditions and other factors and then predicts how likely is this patient to be hospitaliozed in next year. 
- **How Good is it** If you picked patients at random, only about 8 out of every 100 would end up in the hospital. Out of every 100 patients the model flags, about 21 are actually hospitalized, roughly 2.7 times better than picking at random.
- **What it catches:** Of all the patients who will be hospitalized, the model identifies a little more than half (about 5–6 out of every 10).


### Next steps

Given more time, I would have also tried:

- Visualizing the ROC curve and the distribution of predicted probabilities for each class
- Tuning the decision threshold on the predicted probabilities to find the best precision/recall trade-off
- Alternative methods for feature selection
- Evaluating model performance across different patient segments

### AI usage disclaimer

- I built the plan, placeholder code, and functions, then worked with AI to write the code. The ideas, plan, and changes were mine, AI helped me write code faster.
- AI helped me debug the code.
- I wrote all of Summary.md myself without AI help.