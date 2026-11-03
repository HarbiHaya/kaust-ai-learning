# Stage 2 Day 1 — 80 Very Hard MCQs

**Coverage:** ML framework, data problems, generalization, splitting, cross-validation, leakage, regression/classification metrics, and scikit-learn preprocessing workflows.

**Instructions:** Complete the questions closed-book. Choose the single best answer. Target time: 90 minutes. Do not view the answer key until all 80 questions are complete.

## Part A — Learning framework

### 1
A hospital has a stable, legally mandated rule that a patient with a particular laboratory value above a fixed threshold must receive an alert. Which approach is most appropriate for implementing the alert itself?

A. Unsupervised clustering because no target is required  
B. A rule-based system because the decision boundary is explicitly specified  
C. Reinforcement learning because the system takes an action  
D. Supervised learning because historical patients are available

### 2
In a table used to predict apartment rent, each row represents one apartment. The columns are area, district, building age, and monthly rent. Which description is correct?

A. Each column is a sample, and each row is a feature  
B. Monthly rent is a feature because it is numeric  
C. Each row is a sample, the first three named columns are features, and rent is the target  
D. District is the target because it is categorical

### 3
An agent observes a warehouse position, chooses a movement, and receives a penalty for collisions and a reward for delivering an item. Which mapping is correct?

A. Position = action; movement = state; delivery = environment  
B. Position = state; movement = action; delivery reward = reward signal  
C. Position = target; movement = feature; collision = optimizer  
D. Position = parameter; movement = hyperparameter; reward = loss function

### 4
A company has customer behavior data but no churn labels. It wants to identify naturally occurring customer groups. Which method best matches the objective?

A. Regression  
B. Classification  
C. Clustering  
D. Reinforcement learning

### 5
Which pair contains one learned parameter followed by one hyperparameter?

A. A fitted linear-regression coefficient; the regularization strength chosen before fitting  
B. The number of observations; a fitted intercept  
C. Validation accuracy; a predicted probability  
D. A training label; a test feature

### 6
During a forward pass, a neural network primarily:

A. Updates its parameters using gradients  
B. Computes outputs and loss using the current parameter values  
C. Divides the dataset into folds  
D. Calculates only evaluation metrics, not predictions

### 7
The backward pass is responsible for:

A. Computing how the loss changes with respect to model parameters  
B. Selecting the train-test split  
C. Replacing missing values  
D. Converting probabilities into class labels

### 8
Which statement most precisely describes an optimizer?

A. It defines whether a task is regression or classification  
B. It uses gradient information to determine parameter updates  
C. It calculates the validation metric but does not affect training  
D. It creates features from raw data

### 9
A classifier is trained by minimizing cross-entropy but judged using recall because missed positive cases are expensive. Which conclusion is correct?

A. The workflow is invalid because loss and metric must be identical  
B. Cross-entropy controls preprocessing, while recall controls splitting  
C. The loss guides parameter learning, while recall expresses the evaluation priority  
D. Recall must be differentiable for this workflow to be valid

### 10
Which comparison between closed-form and iterative optimization is most accurate?

A. Closed-form methods never use data, while iterative methods do  
B. Closed-form methods directly compute a solution when one is available; iterative methods progressively update parameters  
C. Iterative methods always find the global optimum; closed-form methods may not  
D. Closed-form methods require a validation set; iterative methods do not

### 11
A dataset contains one audio recording per row after each recording has been converted into 40 numerical spectral coefficients. How should the model input be described?

A. The original data modality is audio, while the model receives structured numerical features  
B. The data is still raw audio because the coefficients came from sound  
C. The coefficients are targets because they summarize the recording  
D. The task must be unsupervised because the inputs are numerical

### 12
A model outputs a probability of 0.78 for the positive class. Which statement is necessarily true?

A. The predicted class must be positive  
B. The model has 78% accuracy  
C. The final class depends on the chosen decision threshold  
D. The observation belongs to the positive class

## Part B — Data quality and generalization

### 13
A face-recognition dataset contains one million clear images, but 98% of the people photographed belong to the same demographic group. What is the primary concern?

A. Insufficient sample size  
B. Sampling bias and poor representativeness  
C. Excessive regularization  
D. High optimization error

### 14
A temperature sensor normally records values between 15°C and 45°C. One row contains 500°C because of a hardware failure. This value is best described as:

A. A model parameter  
B. An outlier caused by data-quality noise  
C. A missing value  
D. A hyperparameter

### 15
Missingness in a medical variable occurs mainly when severely ill patients cannot complete a test. Which statement is safest?

A. The missing rows should always be deleted  
B. Missingness may itself carry information related to the target  
C. Mean imputation guarantees an unbiased model  
D. Missingness matters only for unsupervised learning

### 16
Adding thousands of irrelevant random features to a small training dataset is most likely to:

A. Guarantee better generalization because the model receives more information  
B. Increase opportunities for the model to fit accidental patterns  
C. Eliminate variance from the fitted model  
D. Convert a supervised problem into an unsupervised one

### 17
Which operation is feature extraction rather than feature selection?

A. Keeping the 20 original variables with the highest mutual information  
B. Removing columns with near-zero variance  
C. Replacing 100 correlated variables with 10 principal components  
D. Dropping a customer ID column

### 18
A classifier has 58% training accuracy and 57% validation accuracy. The task is balanced, and human performance is approximately 95%. What is the strongest initial diagnosis?

A. High variance  
B. High bias or underfitting  
C. Test-set leakage  
D. Excellent generalization

### 19
A classifier has 99.5% training accuracy and 72% validation accuracy. What is the strongest diagnosis?

A. Underfitting caused by high bias  
B. Overfitting associated with high variance  
C. Perfect generalization  
D. Insufficient optimization because training accuracy is too high

### 20
Models A and B produce the following accuracies: A = 92% training, 84% validation; B = 86% training, 85% validation. Assuming identical validation data and business costs, which statement is best?

A. A is certainly better because its training accuracy is higher  
B. B currently provides stronger evidence of generalization  
C. A is certainly underfitting  
D. B is certainly overfitting

### 21
Training loss keeps decreasing while validation loss decreases initially and then begins rising. The point at which validation loss begins rising most strongly suggests:

A. The model has started fitting training-specific patterns that do not transfer  
B. The data has become balanced  
C. The optimizer has stopped updating parameters  
D. The test set is too large

### 22
A wolf classifier performs well because most wolf images contain snow, while most dog images do not. This is primarily an example of:

A. Closed-form optimization  
B. Shortcut learning using a spurious feature  
C. Stratified sampling  
D. Dimensionality reduction

### 23
Increasing regularization strength usually has which immediate effect?

A. It discourages model complexity, potentially increasing bias while reducing variance  
B. It guarantees both lower bias and lower variance  
C. It creates additional training samples  
D. It prevents the model from making probabilistic predictions

### 24
A dataset contains systematically incorrect labels for 30% of its examples. Which intervention alone is least likely to solve the central problem?

A. Auditing and correcting labels  
B. Improving the label-generation process  
C. Collecting more data using the same flawed labeling process without correction  
D. Measuring agreement between annotators

### 25
A highly flexible model overfits a small dataset. Which change most directly targets model variance without collecting more data?

A. Increase model complexity further  
B. Use a simpler model or stronger regularization  
C. Evaluate repeatedly on the test set  
D. Remove the validation set

### 26
A model trained on daytime road images performs poorly at night. Training and validation images were both sampled only during daytime. This failure is best explained by:

A. A representative validation set proving low variance  
B. Distribution mismatch that the original split failed to reveal  
C. Excessive stratification  
D. A negative R-squared value

### 27
A transaction of SAR 1,000,000 is rare but genuine. Most transactions are below SAR 5,000. What is the most defensible first action?

A. Delete it automatically because it lies far from the mean  
B. Investigate its validity and business meaning before deciding how to handle it  
C. Replace it with the mean without checking  
D. Move it into the test set so it cannot affect training

### 28
Which result provides the strongest evidence of generalization?

A. Near-perfect performance on the training data  
B. Similar performance on representative unseen data that played no role in model selection  
C. A low loss after many training epochs  
D. High performance on examples duplicated across training and validation

## Part C — Splitting, cross-validation, imbalance, and leakage

### 29
Which dataset should influence gradient-based parameter updates?

A. Training set only  
B. Validation set only  
C. Test set only  
D. Validation and test sets together

### 30
A team evaluates 100 models on the same test set and publishes the model with the highest test accuracy. What has happened?

A. The test set has indirectly become part of model selection  
B. Cross-validation variance has been eliminated  
C. The test result has become more unbiased  
D. The model has received more training data

### 31
Which use of the validation set is appropriate?

A. Fitting the final model parameters at every gradient step  
B. Choosing between candidate hyperparameter settings  
C. Reporting the final unbiased generalization result after extensive tuning  
D. Calculating preprocessing statistics before the training split

### 32
Two calls to `train_test_split` use the same data, proportions, and `random_state`. What should normally happen?

A. They produce the same split, supporting reproducibility  
B. They produce different splits because random seeds affect only model training  
C. They guarantee the same model accuracy for every algorithm  
D. They remove sampling bias from the original dataset

### 33
Why is an ordinary random split often inappropriate for forecasting future sales?

A. It may allow later observations to influence training while earlier observations appear in validation  
B. It creates too few numerical features  
C. It guarantees class imbalance  
D. It prevents the use of regression metrics

### 34
A medical dataset contains multiple visits from each patient. Randomly placing visits from the same patient in both training and validation can inflate performance primarily because:

A. The task becomes reinforcement learning  
B. Patient-specific patterns can leak across the split  
C. The number of features decreases  
D. Validation loss becomes a training loss

### 35
In 5-fold cross-validation, how many times is each observation normally used for validation?

A. Zero times  
B. Once  
C. Four times  
D. Five times

### 36
Two models have cross-validation results: A = 0.84 ± 0.01; B = 0.86 ± 0.09. Which conclusion is best?

A. B is unquestionably superior because its mean is higher  
B. A is unquestionably superior because its variability is lower  
C. B has a slightly higher mean but substantially less stable performance across folds  
D. The standard deviation describes training speed, not stability

### 37
The main purpose of stratification during a classification split is to:

A. Scale each feature to unit variance  
B. Preserve approximately the class proportions in each split  
C. Remove minority-class examples  
D. Guarantee equal class counts

### 38
A dataset contains 9,900 negatives and 100 positives. A model predicts every observation as negative. Which statement is correct?

A. Accuracy is 99%, while positive-class recall is 0%  
B. Accuracy and recall are both 99%  
C. Precision is 100% because the model makes no false-positive predictions  
D. Specificity is 0%

### 39
A researcher standardizes all features using the complete dataset and then performs a train-test split. Why is this problematic?

A. Standardization cannot be used with classification  
B. Test-set statistics influence the transformed training representation  
C. Standardization changes labels into features  
D. The split will no longer contain the same number of rows

### 40
Which workflow avoids leakage when imputing missing values?

A. Calculate imputation values on all data, then split  
B. Split first, fit the imputer on training data, and apply it unchanged to validation/test data  
C. Fit a different imputer independently on every test row  
D. Delete all rows containing missing data before examining the dataset

### 41
A team selects the 50 features most correlated with the target using the full dataset, then evaluates with cross-validation. What is the main issue?

A. Feature selection occurred outside the folds and used validation-fold target information  
B. Correlation cannot be used for feature selection  
C. Cross-validation works only with all original features  
D. The model will necessarily underfit

### 42
Which operation may be safely calculated using both training and test datasets without leaking predictive information?

A. The global mean used for imputation  
B. The global standard deviation used for scaling  
C. The list of column names required to verify schema compatibility, without examining their values or targets  
D. The correlation of every feature with the target

### 43
A feature called `days_until_account_closed` is used to predict whether an account will close. Why is it suspicious?

A. It is numerical  
B. It likely contains information that becomes available only after or near the predicted event  
C. It requires scaling  
D. It will always cause underfitting

### 44
During cross-validation, a scaler is fitted once on the entire training dataset before folds are created. What is the correct interpretation?

A. There is no issue because the held-out test set remains untouched  
B. Validation folds influence the scaling statistics used in their corresponding training folds  
C. Scaling never creates leakage  
D. Cross-validation automatically refits every object created outside it

### 45
After hyperparameters are chosen using cross-validation on the training data, what is the best final procedure?

A. Report the highest fold score as the final test score  
B. Fit the selected workflow on the full training data, then evaluate once on the untouched test set  
C. Continue selecting models based on repeated test evaluations  
D. Add the test labels to training before evaluation

## Part D — Metrics and decision thresholds

For Questions 46–50, use: **TP = 36, FP = 12, TN = 943, FN = 9**.

### 46
What is the accuracy?

A. 95.5%  
B. 97.9%  
C. 98.8%  
D. 80.0%

### 47
What is the positive-class precision?

A. 75.0%  
B. 80.0%  
C. 97.9%  
D. 98.7%

### 48
What is the positive-class recall?

A. 75.0%  
B. 77.4%  
C. 80.0%  
D. 98.7%

### 49
What is the specificity?

A. Approximately 75.0%  
B. Approximately 80.0%  
C. Approximately 97.9%  
D. Approximately 98.7%

### 50
What is the F1 score?

A. Approximately 0.774  
B. Approximately 0.800  
C. Approximately 0.979  
D. Approximately 0.987

### 51
For a fixed probabilistic classifier, increasing the threshold required to predict the positive class will usually:

A. Increase both recall and the number of positive predictions  
B. Decrease recall and may increase precision  
C. Leave the confusion matrix unchanged  
D. Increase false positives while decreasing true negatives

### 52
In a screening system where a missed disease case is far more costly than an unnecessary follow-up, which priority is most defensible?

A. Maximize specificity regardless of recall  
B. Emphasize recall while monitoring the resulting false-positive burden  
C. Use accuracy alone  
D. Maximize the decision threshold

### 53
An email provider wants to avoid placing legitimate emails in spam. Which positive-class metric deserves special attention if “spam” is the positive class?

A. Precision, because false-positive spam predictions should be limited  
B. Recall, because false negatives are legitimate emails  
C. Specificity, because it ignores true negatives  
D. RMSE, because probabilities are continuous

### 54
A regression dataset contains a few extremely large target errors that the business considers especially costly. Which metric penalizes those errors most strongly?

A. MAE  
B. MSE  
C. Classification accuracy  
D. Specificity

### 55
Why is RMSE often easier to interpret than MSE?

A. RMSE is always smaller than MAE  
B. RMSE has the same units as the target variable  
C. RMSE ignores large errors  
D. RMSE ranges only from 0 to 1

### 56
If prediction errors are measured in Saudi riyals, the units of MSE are:

A. Riyals  
B. Squared riyals  
C. No units  
D. Percent

### 57
A regression model evaluated on a test set has R² = −0.4. Which interpretation is best?

A. The model explains 40% of the target variance  
B. The model performs worse under squared error than predicting the test-set target mean as a constant baseline  
C. R² cannot be negative, so the software is broken  
D. The model's predictions are negatively correlated with every feature

### 58
Which condition yields R² = 1 on an evaluated dataset?

A. Predictions equal the observed targets for every evaluated sample  
B. The mean residual is zero  
C. The model predicts the target mean for every sample  
D. MAE and RMSE have equal values

### 59
For an extremely imbalanced dataset with rare positives, why is a precision-recall curve often more informative than a ROC curve?

A. Precision-recall analysis focuses directly on positive predictions and recovered positives  
B. ROC curves cannot be computed for binary classification  
C. Precision-recall curves do not depend on thresholds  
D. ROC-AUC always equals accuracy

### 60
A classifier ranks every positive example above every negative example, but the currently selected threshold produces poor recall. Which statement can still be true?

A. ROC-AUC can be 1.0 despite poor performance at the selected threshold  
B. ROC-AUC must equal recall  
C. The classifier cannot produce probabilities  
D. Precision must be zero at every threshold

### 61
In a three-class problem, one rare class performs very poorly while two common classes perform well. Which averaging method is most likely to expose the rare class problem?

A. Macro averaging  
B. Weighted averaging  
C. Overall accuracy only  
D. Micro averaging only

### 62
For ordinary single-label multiclass classification, micro-averaged recall is typically equal to:

A. Overall accuracy  
B. Macro recall  
C. The recall of the rarest class  
D. R²

### 63
For residual defined as `actual − predicted`, an actual value of 120 and prediction of 135 produce:

A. Residual = 15, an underprediction  
B. Residual = −15, an overprediction  
C. Residual = −15, an underprediction  
D. Residual = 15, an overprediction

### 64
Which statement about metric selection is strongest?

A. Accuracy should be used for every classification problem because it is intuitive  
B. The preferred metric should reflect the relative costs of the errors and the class distribution  
C. F1 is always superior to precision and recall  
D. ROC-AUC determines the correct business threshold automatically

### 65
Two fraud models have equal recall. Model A has precision 0.20, and Model B has precision 0.40. Which statement is correct?

A. B produces fewer false alarms per recovered fraud case  
B. A necessarily has higher accuracy  
C. Both models have identical confusion matrices  
D. Precision cannot be compared when recall is equal

## Part E — Scikit-learn and preprocessing workflows

### 66
For a scikit-learn estimator, calling `fit(X_train, y_train)` primarily:

A. Learns estimator state or parameters from the training data  
B. Produces final test predictions without learning  
C. Guarantees calibrated probabilities  
D. Splits the data into folds

### 67
For a fitted scaler, calling `transform(X_test)` should:

A. Recalculate scaling statistics from `X_test`  
B. Apply the statistics previously learned during `fit`  
C. Fit the downstream classifier  
D. Convert regression targets into classes

### 68
Which sequence is correct?

A. `fit_transform(X_train)`, then `transform(X_test)`  
B. `fit_transform(X_train)`, then `fit_transform(X_test)`  
C. `transform(X_train)`, then `fit(X_test)`  
D. `predict(X_train)`, then `fit_transform(X_test)`

### 69
Which statement about `StandardScaler` is correct?

A. It normally centers using the fitted mean and scales using the fitted standard deviation  
B. It maps every feature exactly into [0, 1]  
C. It uses the median and interquartile range  
D. It removes all outliers

### 70
Why can `MinMaxScaler` be strongly affected by an extreme outlier?

A. Its fitted minimum and maximum determine the scaling range for other values  
B. It converts all values into categories  
C. It ignores the training data distribution  
D. It always uses the target variable

### 71
Which scaler is designed to be more resistant to extreme values by using robust statistics?

A. `StandardScaler`  
B. `MinMaxScaler`  
C. `RobustScaler`  
D. `OneHotEncoder`

### 72
Which variable is most appropriate for ordinal encoding?

A. City names with no inherent order  
B. Customer IDs  
C. Satisfaction levels: low, medium, high  
D. Random invoice codes

### 73
Why is independently running one-hot encoding on training and test data dangerous?

A. The sets or orders of generated columns may differ  
B. One-hot encoding works only on numerical columns  
C. Test data must never contain categorical values  
D. It always causes high bias

### 74
A category appears in the test set but was absent from training. Which configuration most directly prevents `OneHotEncoder` from failing on that category?

A. `handle_unknown="ignore"`  
B. `random_state=0`  
C. `with_mean=False`  
D. `strategy="median"`

### 75
A dataset contains numerical columns needing imputation and scaling, plus categorical columns needing imputation and one-hot encoding. Which tool coordinates different transformations for different columns?

A. `DummyClassifier`  
B. `ColumnTransformer`  
C. `train_test_split`  
D. `confusion_matrix`

### 76
What is the most important correctness benefit of placing preprocessing and a model in one `Pipeline`?

A. It guarantees the selected model is optimal  
B. It helps apply the same ordered transformations and allows fold-specific preprocessing during cross-validation  
C. It eliminates the need for a test set  
D. It makes all features normally distributed

### 77
Which cross-validation call is safest when scaling is required?

A. Scale the complete dataset first, then cross-validate only the classifier  
B. Pass a pipeline containing the scaler and classifier to `cross_val_score`  
C. Fit the scaler on the test set, then cross-validate the classifier  
D. Use a different manually selected scaler for each validation result

### 78
A `DummyClassifier` scores 91% accuracy, while a complex model scores 92% on a highly imbalanced task. What is the best conclusion?

A. The complex model is clearly excellent  
B. Accuracy shows only a small improvement over a weak baseline, so class-sensitive metrics and error costs must be examined  
C. The dummy model has learned nonlinear patterns  
D. Baselines are irrelevant once a complex model is trained

### 79
Setting `random_state=42` in every applicable component primarily supports:

A. Reproducibility of randomized operations, not guaranteed model quality  
B. Automatic elimination of bias  
C. Perfect test accuracy  
D. Selection of 42 model parameters

### 80
Which workflow is the most defensible end-to-end procedure for a tabular classification problem?

A. Inspect all data, fit preprocessing on all data, split, tune on test, report the best test result  
B. Separate an untouched test set, place learned preprocessing and the model in a pipeline, tune with cross-validation on training data, refit the selected pipeline on all training data, and evaluate once on test  
C. Split data, independently preprocess each subset, repeatedly tune using test accuracy, and report training accuracy  
D. Fit the model before preprocessing, use the validation set for parameter updates, and use the training set for final evaluation

---

## Answer key

1. B  
2. C  
3. B  
4. C  
5. A  
6. B  
7. A  
8. B  
9. C  
10. B  
11. A  
12. C  
13. B  
14. B  
15. B  
16. B  
17. C  
18. B  
19. B  
20. B  
21. A  
22. B  
23. A  
24. C  
25. B  
26. B  
27. B  
28. B  
29. A  
30. A  
31. B  
32. A  
33. A  
34. B  
35. B  
36. C  
37. B  
38. A  
39. B  
40. B  
41. A  
42. C  
43. B  
44. B  
45. B  
46. B  
47. A  
48. C  
49. D  
50. A  
51. B  
52. B  
53. A  
54. B  
55. B  
56. B  
57. B  
58. A  
59. A  
60. A  
61. A  
62. A  
63. B  
64. B  
65. A  
66. A  
67. B  
68. A  
69. A  
70. A  
71. C  
72. C  
73. A  
74. A  
75. B  
76. B  
77. B  
78. B  
79. A  
80. B

## Mastery thresholds

- **72–80:** Strong Day 1 mastery. Proceed to the labs and blank reconstruction.
- **64–71:** Good foundation, but repair every incorrect concept before the exercise.
- **52–63:** Partial understanding. Revisit the related slide sections and repeat missed questions after 24 hours.
- **Below 52:** Repeat Day 1 theory in smaller sections before moving to the exercise.

Regardless of the total score, do not treat Day 1 as complete if you miss more than two questions from Questions 29–45 or more than three from Questions 46–65. Splitting, leakage, and metric selection are high-priority exam concepts.
