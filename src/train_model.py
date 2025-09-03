import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# Paths
data_path = os.path.join("data", "dataset.csv")
model_path = os.path.join("models", "doctor.pkl")

# Load dataset
df = pd.read_csv(data_path)

# Fill NaN with 'None'
df.fillna('None', inplace=True)

# Combine symptoms into one list per row
df['Symptoms'] = df[df.columns[1:]].values.tolist()

# Encode symptoms as strings for model input
df['Symptoms'] = df['Symptoms'].apply(lambda x: ','.join(sorted(list(set(x) - {'None'}))))

# Encode diseases
le = LabelEncoder()
df['DiseaseEncoded'] = le.fit_transform(df['Disease'])

# Features and labels
X = df['Symptoms']
y = df['DiseaseEncoded']

# Convert symptoms into numeric tokens
all_symptoms = sorted(set(','.join(X).split(',')))
symptom_to_num = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

def encode_symptoms(symptom_list):
    vector = [0] * len(symptom_to_num)
    for s in symptom_list.split(','):
        if s in symptom_to_num:
            vector[symptom_to_num[s]] = 1
    return vector

X_encoded = pd.DataFrame(X.apply(encode_symptoms).tolist(), columns=all_symptoms)

# Train model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Save model and encoders
os.makedirs('models', exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'encoder': le, 'symptom_map': symptom_to_num}, f)

print("✅ Model trained and saved at models/doctor.pkl")
