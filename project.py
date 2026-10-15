

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")


# ============================================================
# PART 1: LOAD AND UNDERSTAND THE PROBLEM
# ============================================================

print("\n" + "=" * 70)
print("PART 1: LOAD AND UNDERSTAND THE PROBLEM")
print("=" * 70)

# Find CSV in the same folder as this Python file
try:
    file_path = os.path.join(os.path.dirname(__file__), "telcom-churn.csv")
except NameError:
    file_path = "telecom-churn.csv"

df = pd.read_csv(r"C:\Users\Guna\New folder\telecom-churn.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nProblem Type: CLASSIFICATION")
print("X (Features): tenure_months, monthly_charges, complaints")
print("y (Target): churn")
print("churn = 1 means customer left")
print("churn = 0 means customer stayed")


# ============================================================
# PART 2: EXPLORE BEFORE YOU MODEL
# ============================================================

print("\n" + "=" * 70)
print("PART 2: EXPLORE BEFORE YOU MODEL")
print("=" * 70)

churn_rate = df["churn"].mean() * 100

print(f"\nChurn rate: {churn_rate:.2f}%")
print(f"Customers who churned: {df['churn'].sum()}")
print(f"Customers who stayed: {(df['churn'] == 0).sum()}")

print("\nAverage values by churn:")
comparison = df.groupby("churn").mean()
print(comparison)

print("\nCorrelation matrix:")
correlation = df.corr()
print(correlation)

# Histograms
plt.figure(figsize=(8, 5))
plt.hist(df["tenure_months"], bins=20, edgecolor="black")
plt.title("Distribution of Tenure Months")
plt.xlabel("Tenure Months")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["monthly_charges"], bins=20, edgecolor="black")
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("\nFinding:")
print("Monthly charges and complaints have strong positive relationships with churn.")
print("Tenure has a weaker relationship with churn compared with the other features.")


# ============================================================
# PART 3: SPLIT THEN SCALE
# ============================================================

print("\n" + "=" * 70)
print("PART 3: SPLIT THEN SCALE")
print("=" * 70)

X = df[["tenure_months", "monthly_charges", "complaints"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

print("\nWhy stratify?")
print("Stratify keeps approximately the same churn/non-churn proportion")
print("in both training and testing datasets.")

print("\nWhy fit scaler only on training data?")
print("To prevent information from the test dataset leaking into the model.")


# ============================================================
# PART 4: BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("PART 4: LOGISTIC REGRESSION")
print("=" * 70)

logreg = LogisticRegression()

logreg.fit(X_train_s, y_train)

pred = logreg.predict(X_test_s)
prob = logreg.predict_proba(X_test_s)[:, 1]

coefficients = logreg.coef_[0]
odds_ratios = np.exp(coefficients)

odds_table = pd.DataFrame({
    "feature": X.columns,
    "coefficient": coefficients,
    "odds_ratio": odds_ratios
})

print("\nLogistic Regression Coefficients and Odds Ratios:")
print(odds_table.to_string(index=False))

print("\nBusiness interpretation:")

for feature, odds in zip(X.columns, odds_ratios):
    if odds > 1:
        print(
            f"- {feature}: A 1-standard-deviation increase is associated "
            f"with higher churn odds by about {(odds - 1) * 100:.2f}%."
        )
    else:
        print(
            f"- {feature}: A 1-standard-deviation increase is associated "
            f"with lower churn odds."
        )


# ============================================================
# PART 5: SCORE THE MODEL PROPERLY
# ============================================================

print("\n" + "=" * 70)
print("PART 5: MODEL EVALUATION")
print("=" * 70)

cm = confusion_matrix(y_test, pred)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Details:")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")

print("\nClassification Report:")
print(classification_report(y_test, pred))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1 = 2 * precision * recall / (precision + recall)

print("\nMetrics calculated manually:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

auc = roc_auc_score(y_test, prob)

print(f"\nAUC: {auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Nobody churns baseline
baseline_accuracy = (y_test == 0).mean()

print(f"\nAccuracy if model predicts nobody churns: {baseline_accuracy:.4f}")
print(
    f"That is {baseline_accuracy * 100:.2f}%. "
    "This shows why accuracy alone can be misleading."
)

if fn > fp:
    print("\nThe model makes more False Negatives than False Positives.")
    print("This matters because missing a customer who will churn can lose valuable revenue.")


# ============================================================
# PART 6: CHOOSE THE BEST THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("PART 6: THRESHOLD ANALYSIS")
print("=" * 70)

thresholds_to_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

rows = []

for th in thresholds_to_test:

    pred_th = (prob >= th).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred_th
    ).ravel()

    precision_th = precision_score(
        y_test,
        pred_th,
        zero_division=0
    )

    recall_th = recall_score(
        y_test,
        pred_th,
        zero_division=0
    )

    f1_th = f1_score(
        y_test,
        pred_th,
        zero_division=0
    )

    # Original business assumption
    net_value = 4000 * tp - 300 * (tp + fp)

    # Alternative offer cost
    net_value_2000 = 4000 * tp - 2000 * (tp + fp)

    rows.append({
        "threshold": th,
        "precision": precision_th,
        "recall": recall_th,
        "f1": f1_th,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "net_value_₹300": net_value,
        "net_value_₹2000": net_value_2000
    })

threshold_df = pd.DataFrame(rows)

print("\nThreshold Results:")
print(threshold_df.to_string(index=False))

best_300 = threshold_df.loc[
    threshold_df["net_value_₹300"].idxmax()
]

best_2000 = threshold_df.loc[
    threshold_df["net_value_₹2000"].idxmax()
]

print("\nRecommendation when offer costs ₹300:")
print(
    f"Threshold = {best_300['threshold']:.1f}, "
    f"Net Value = ₹{best_300['net_value_₹300']:,.0f}"
)

print("\nRecommendation when offer costs ₹2,000:")
print(
    f"Threshold = {best_2000['threshold']:.1f}, "
    f"Net Value = ₹{best_2000['net_value_₹2000']:,.0f}"
)

print("\nBusiness reasoning:")
print(
    "When the offer costs ₹300, the model should use the threshold "
    "that gives the highest expected net value."
)
print(
    "If the offer becomes more expensive, the company should be more selective "
    "and contact fewer customers."
)


# ============================================================
# PART 7: COMPARE SIX MODELS
# ============================================================

print("\n" + "=" * 70)
print("PART 7: COMPARE SIX MODELS")
print("=" * 70)

models = {
    "Logistic": LogisticRegression(),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Tree (d=3)": DecisionTreeClassifier(
        max_depth=3,
        random_state=1
    ),

    "Random forest": RandomForestClassifier(
        n_estimators=200,
        random_state=1
    ),

    "SVM": SVC(
        random_state=1
    )
}

results = []

for name, model in models.items():

    model.fit(X_train_s, y_train)

    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)

    train_accuracy = model.score(
        X_train_s,
        y_train
    )

    test_accuracy = model.score(
        X_test_s,
        y_test
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Test F1": test_f1
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test F1",
    ascending=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

winner = results_df.iloc[0]["Model"]

print(f"\nWinner based on Test F1: {winner}")

print("\nDecision Tree Rules:")
tree_model = models["Tree (d=3)"]

print(
    export_text(
        tree_model,
        feature_names=list(X.columns)
    )
)

print("\nOverfitting check:")

for _, row in results_df.iterrows():

    if row["Train Accuracy"] == 1.0:
        print(
            f"{row['Model']} has 1.0 training accuracy. "
            "This can indicate overfitting."
        )


# ============================================================
# PART 8: CROSS-VALIDATE THE WINNER
# ============================================================

print("\n" + "=" * 70)
print("PART 8: CROSS-VALIDATION")
print("=" * 70)

# Scale complete dataset
full_scaler = StandardScaler()
X_full_scaled = full_scaler.fit_transform(X)

# Create the winning model again
if winner == "Logistic":
    winning_model = LogisticRegression()

elif winner == "KNN":
    winning_model = KNeighborsClassifier(n_neighbors=5)

elif winner == "Naive Bayes":
    winning_model = GaussianNB()

elif winner == "Tree (d=3)":
    winning_model = DecisionTreeClassifier(
        max_depth=3,
        random_state=1
    )

elif winner == "Random forest":
    winning_model = RandomForestClassifier(
        n_estimators=200,
        random_state=1
    )

elif winner == "SVM":
    winning_model = SVC(random_state=1)

cv_scores = cross_val_score(
    winning_model,
    X_full_scaled,
    y,
    cv=5,
    scoring="f1"
)

print("\nFive Cross-Validation F1 Scores:")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

cv_mean = cv_scores.mean()
cv_min = cv_scores.min()
cv_max = cv_scores.max()
cv_spread = cv_max - cv_min

print(f"\nMean F1: {cv_mean:.4f}")
print(f"Minimum F1: {cv_min:.4f}")
print(f"Maximum F1: {cv_max:.4f}")
print(f"Spread: {cv_spread:.4f}")

print("\nInterpretation:")
if cv_spread < 0.10:
    print("The model is relatively stable across folds.")
else:
    print(
        "The model has noticeable variation across folds, "
        "so performance depends somewhat on which rows it sees."
    )


# ============================================================
# PART 9: FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("PART 9: FINAL RECOMMENDATION")
print("=" * 70)

# Best threshold for ₹300
recommended_threshold = best_300["threshold"]

recommended_pred = (
    prob >= recommended_threshold
).astype(int)

recommended_customers = recommended_pred.sum()

recommended_tp = int(best_300["TP"])
recommended_fp = int(best_300["FP"])
recommended_fn = int(best_300["FN"])
recommended_value = int(best_300["net_value_₹300"])

print("\nFINAL BUSINESS RECOMMENDATION")
print("-" * 50)

print("\nFinding 1:")
print(
    f"{churn_rate:.2f}% of the customers churned "
    f"({int(df['churn'].sum())} out of {len(df)} customers)."
)

print("\nFinding 2:")
print(
    f"Monthly charges and complaints are the strongest positive "
    f"churn-related features in the dataset."
)

print("\nFinding 3:")
print(
    f"The Logistic Regression model achieved "
    f"{accuracy * 100:.2f}% test accuracy, "
    f"{f1:.4f} F1 score and {auc:.4f} AUC."
)

print("\nWho should the company contact?")
print(
    f"Using threshold {recommended_threshold:.1f}, "
    f"the company should contact approximately "
    f"{recommended_customers} customers."
)

print(
    f"Expected True Positives: {recommended_tp}"
)

print(
    f"Expected False Positives: {recommended_fp}"
)

print(
    f"Expected False Negatives: {recommended_fn}"
)

print(
    f"Estimated net value at ₹300 offer cost: "
    f"₹{recommended_value:,}"
)

print("\nRecommended actions:")

print(
    "1. Pay special attention to customers with higher monthly charges, "
    "because higher charges are strongly associated with higher churn odds."
)

print(
    "2. Monitor customers who raise more complaints and provide faster "
    "customer-service support or retention offers to reduce churn risk."
)

print("\nLimitation:")
print(
    "This dataset contains only tenure, monthly charges and complaints. "
    "It cannot tell us whether other factors such as service quality, "
    "competitor offers, customer satisfaction, contract type or demographics "
    "are causing customers to churn."
)

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)