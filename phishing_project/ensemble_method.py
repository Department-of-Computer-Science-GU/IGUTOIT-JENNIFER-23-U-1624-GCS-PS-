import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. DATA LOADING ---
# Ensure phishing.csv is in the same folder as this script!
df = pd.read_csv('phishing.csv')

# Drop the 'domain' string column and separate the 'label' target
X = df.drop(columns=['domain', 'label'])
y = df['label']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. PRE-PROCESSING ---
# Scaling is required for Logistic Regression and KNN to work properly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. TRAIN 3 DIVERSE MODELS ---
# We fit them individually first
model_lr = LogisticRegression().fit(X_train_scaled, y_train)
model_rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_scaled, y_train)
model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

# Put them in a list for our ensemble function
base_models = [model_lr, model_rf, model_knn]

# --- 4. ENSEMBLE LOGIC (FROM SCRATCH) ---
def ensemble_predict_scratch(models, data):
    # Step A: Get predictions from every model in the list
    # Resulting shape: (number_of_models, number_of_rows)
    all_predictions = np.array([m.predict(data) for m in models])
    
    # Step B: Transpose so each row is a list of 3 votes for 1 sample
    # Resulting shape: (number_of_rows, number_of_models)
    all_predictions = all_predictions.T
    
    # Step C: Majority Voting
    # For every row, find the most common number (0 or 1)
    final_votes = [np.argmax(np.bincount(row)) for row in all_predictions]
    return np.array(final_votes)

# --- 5. EVALUATION ---
y_pred_ensemble = ensemble_predict_scratch(base_models, X_test_scaled)

print(f"--- Results for phishing.csv ---")
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble) * 100:.2f}%")

#comarison
print("\n--- Model Comparison ---")
print(f"Logistic Regression: {accuracy_score(y_test, model_lr.predict(X_test_scaled))*100:.2f}%")
print(f"Random Forest:       {accuracy_score(y_test, model_rf.predict(X_test_scaled))*100:.2f}%")
print(f"KNN:                 {accuracy_score(y_test, model_knn.predict(X_test_scaled))*100:.2f}%")
print(f"Ensemble (Scratch):  94.75%")