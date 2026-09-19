import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearns.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier   
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV   

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

df.head()

df["Class"].value_counts(normalize=True)

df["Amount_log"] = np.log1p(df["Amount"])

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42
                                                    )   

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

y_probs = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_probs))

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Under-sampling
fraudes = df[df["Class"] == 1]
nao_fraudes = df[df["Class"] == 0].sample(n=len(fraudes), random_state=42)
df_under = pd.concat([fraudes, nao_fraudes])

#Over-sampling
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

rf = RandomForestClassifier(
n_estimators=50, 
max_depth=10,
class_weight="balanced",
n_jobs=-1,
random_state=42
                           )

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test, y_pred_rf))

Pipeline =  Pipeline([
("scaler", StandardScaler()),
("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train, y_train)
y_pred_pipeline = pipeline.predict(x_test)

thresholds = 0.3

y_pred_custom = (y_probs >= thresholds).astype(int)
print(classification_report(y_test, y_pred_custom))

xgb = XGBClassifier(
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

bst.update(dtrain, iteration=i, fobj=obj)

importance = xgb.feature_importances_

plt.bar(range(len(importance)), importance)
plt.title("Feature Importance")
plt.show()

param_grid = {
    "max_depth": [3, 5, ],
    "n_estimators": [50, 100]
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid,
    scoring="recall",
    cv=3
)

grid.fit(x_train, y_train)

print("Best parameters:", grid_search.best_params_)

explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])

shap.plots.bar(shap_values)

