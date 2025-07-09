# pip install AI Fairness 360
# pip install aif360
# fairness_Team4_COMPAS_Audit_Code_FIXED.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# ✅ Load local CSV
df = pd.read_csv("compas-scores-two-years.csv")

# ✅ Basic preprocessing
df['race_binary'] = df['race'].apply(lambda x: 1 if x.strip() == 'Caucasian' else 0)

label_col = 'two_year_recid'
protected_col = 'race_binary'
features = ['age', 'priors_count', protected_col]

X = df[features]
y = df[label_col]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# ✅ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ✅ Wrap in BinaryLabelDataset with required args
test_bld = BinaryLabelDataset(
    df=pd.DataFrame(X_test, columns=features).assign(label=y_test.values),
    label_names=['label'],
    protected_attribute_names=[protected_col]
)

pred_bld = BinaryLabelDataset(
    df=pd.DataFrame(X_test, columns=features).assign(label=y_pred),
    label_names=['label'],
    protected_attribute_names=[protected_col]
)

# ✅ Fairness metrics
privileged_groups = [{protected_col: 1}]
unprivileged_groups = [{protected_col: 0}]

metric = ClassificationMetric(
    test_bld, pred_bld,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

spd = metric.statistical_parity_difference()
eod = metric.equal_opportunity_difference()
fpr_u = metric.false_positive_rate(privileged=False)
fpr_p = metric.false_positive_rate(privileged=True)

plt.bar(['African-American', 'Caucasian'], [fpr_u, fpr_p], color=['red', 'blue'])
plt.title("False Positive Rate by Race")
plt.ylabel("False Positive Rate")
plt.ylim(0, 1)
plt.show()

# ✅ Reweighing
train_bld = BinaryLabelDataset(
    df=pd.DataFrame(X_train, columns=features).assign(label=y_train.values),
    label_names=['label'],
    protected_attribute_names=[protected_col]
)

RW = Reweighing(unprivileged_groups, privileged_groups)
train_bld_rw = RW.fit_transform(train_bld)

X_train_rw = scaler.fit_transform(train_bld_rw.features)
y_train_rw = train_bld_rw.labels.ravel()

model_rw = LogisticRegression(solver='liblinear')
model_rw.fit(X_train_rw, y_train_rw)
y_pred_rw = model_rw.predict(X_test_scaled)

pred_bld_rw = BinaryLabelDataset(
    df=pd.DataFrame(X_test, columns=features).assign(label=y_pred_rw),
    label_names=['label'],
    protected_attribute_names=[protected_col]
)

metric_rw = ClassificationMetric(
    test_bld, pred_bld_rw,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\nBias Metrics BEFORE Reweighing:")
print(f"Statistical Parity Difference: {spd:.3f}")
print(f"Equal Opportunity Difference: {eod:.3f}")
print(f"False Positive Rate (African-American): {fpr_u:.3f}")
print(f"False Positive Rate (Caucasian): {fpr_p:.3f}")
print(f"Accuracy Before: {accuracy_score(y_test, y_pred):.3f}")

print("\nBias Metrics AFTER Reweighing:")
print(f"Statistical Parity Difference: {metric_rw.statistical_parity_difference():.3f}")
print(f"Equal Opportunity Difference: {metric_rw.equal_opportunity_difference():.3f}")
print(f"False Positive Rate (African-American): {metric_rw.false_positive_rate(privileged=False):.3f}")
print(f"False Positive Rate (Caucasian): {metric_rw.false_positive_rate(privileged=True):.3f}")
print(f"Accuracy After: {accuracy_score(y_test, y_pred_rw):.3f}")
