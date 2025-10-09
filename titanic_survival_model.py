import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# Titanic Survival Prediction Script
# ===============================

# Load dataset
data = pd.read_csv("tested.csv")

# Ensure proper column names
data.columns = data.columns.str.strip()

# Handle missing values
data = data.fillna(data.median(numeric_only=True))

# Encode categorical if exists (like Sex)
if 'Sex' in data.columns and data['Sex'].dtype == 'object':
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

# Keep only numeric columns for analysis
numeric_data = data.select_dtypes(include=['number'])

# Split features and target
X = numeric_data.drop(columns=['Survived'])
y = numeric_data['Survived']

# Split into train, validation, and test (70/15/15)
x_temp, x_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# === Evaluate on Validation Set ===
y_val_pred = model.predict(x_val)
y_val_probs = model.predict_proba(x_val)[:, 1] #pick 2nd column
''' It gives probabilities for each class, e.g. 
[[0.83, 0.17],
 [0.12, 0.88],
 [0.45, 0.55]]
 where:
First column = probability of class 0 (did not survive)
Second column = probability of class 1 (survived)
'''

metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred),
    "Recall": recall_score(y_val, y_val_pred),
    "F1 Score": f1_score(y_val, y_val_pred)
}

print("\n===== MODEL PERFORMANCE ON VALIDATION SET =====")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

# === Evaluate on Test Set ===
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy (Unseen Data): {test_accuracy:.3f}")

# === Visualizations ===
plt.figure(figsize=(8,6))
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
feature_importance.plot(kind='barh')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# === Predict survival likelihoods (for demonstration) ===
print("\nUsing validation dataset for demonstration of likely survivors...")
predictions = pd.DataFrame({
    'Predicted_Probability': y_val_probs,
    'Actual_Survived': y_val.reset_index(drop=True) #since we randomly selected data, this ensures pandas does not try to arrange by indexes which might cause mismatch issues
})
predictions['Prediction_Label'] = predictions['Predicted_Probability'].apply(
    lambda p: 'Likely to Survive' if p >= 0.5 else 'Likely Not to Survive'
)

print("\n===== SAMPLE PREDICTIONS =====")
print(predictions.head(10))

print("\n===== SURVIVAL SUMMARY (For our samples) =====")
print(predictions['Prediction_Label'].value_counts())
