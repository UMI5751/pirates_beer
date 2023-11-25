#te

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report


df_adjusted_preferences = pd.read_csv('adjusted_beer_preferences.csv')


encoder = LabelEncoder()
df_adjusted_preferences['Target Style Encoded'] = encoder.fit_transform(df_adjusted_preferences['Target Style'])


X_adjusted = df_adjusted_preferences[['Age', 'Average Rating', 'Alcohol Preference', 'Bitterness Preference', 'Color Preference']]
y_adjusted = df_adjusted_preferences['Target Style Encoded']


X_train_adjusted, X_test_adjusted, y_train_adjusted, y_test_adjusted = train_test_split(X_adjusted, y_adjusted, test_size=0.2, random_state=0)


rf_clf_finetuned = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=0)


rf_clf_finetuned.fit(X_train_adjusted, y_train_adjusted)


y_pred_finetuned = rf_clf_finetuned.predict(X_test_adjusted)


accuracy_finetuned = accuracy_score(y_test_adjusted, y_pred_finetuned)

report = classification_report(y_test_adjusted, y_pred_finetuned)
print(report)

model_file_path = 'beer_recommendation_model_finetuned.joblib'
joblib.dump(rf_clf_finetuned, model_file_path)



