
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from enhanced_feature_extractor import extract_enhanced_features

df = pd.read_csv("updated_training_data.csv")
features = extract_enhanced_features(df)
features['is_spam'] = df['is_spam']

X = features.drop(columns=['URL', 'url', 'domain', 'is_spam'], errors='ignore')
y = features['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("✅ Accuracy:", model.score(X_test, y_test))
joblib.dump(model, "spam_model_v2.pkl")
print("📦 Model saved to spam_model_v2.pkl")
