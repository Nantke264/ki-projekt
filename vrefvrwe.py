"""
Example Python script to train a neural network to predict house price
based on various features such as area, bedrooms, bathrooms, etc.

Make sure you have installed:
    - pandas
    - numpy
    - scikit-learn
    - tensorflow (or keras)

pip install pandas numpy scikit-learn tensorflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score  # <-- Import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------------------------------------------
# 1. Read the CSV data
# -------------------------------------------------------------------
df = pd.read_csv('Housing.csv')

# Identify numeric columns to add noise (e.g., area, bedrooms, bathrooms, stories, parking, price)
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']

# Number of times to duplicate
dup_factor = 5

augmented_rows = []
for idx, row in df.iterrows():
    for _ in range(dup_factor):
        new_row = row.copy()
        for col in numeric_cols:
            # For each numeric column, add small Gaussian noise (1% of its value)
            noise = np.random.normal(0, 0.01 * row[col])
            new_row[col] = row[col] + noise
        augmented_rows.append(new_row)

augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)

# Now combine with original data
df_augmented = pd.concat([df, augmented_df], ignore_index=True)

# Make sure to fix integer or categorical columns if they must stay integer (round them, clip them, etc.)
df_augmented['bedrooms'] = df_augmented['bedrooms'].round().clip(lower=0)

print("Original dataset size:", len(df))
print("Augmented dataset size:", len(df_augmented))

# -------------------------------------------------------------------
# 2. Preprocess / Clean data
# -------------------------------------------------------------------
yes_no_columns = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

for col in yes_no_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode 'furnishingstatus'
data = pd.get_dummies(df, columns=['furnishingstatus'])

# Separate features (X) and target (y)
X = data.drop('price', axis=1).values
y = data['price'].values

# -------------------------------------------------------------------
# 3. Split into train and test
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------------------------
# 4. Build the neural network model
# -------------------------------------------------------------------
model = Sequential()
n_features = X_train.shape[1]

model.add(Dense(20, activation='relu', input_shape=(n_features,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))  # Single output

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# -------------------------------------------------------------------
# 5. Train the model
# -------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=500,
    batch_size=16,
    verbose=1
)

# -------------------------------------------------------------------
# 6. Evaluate on the test set
# -------------------------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE (loss): {test_loss:.2f}")
print(f"Test MAE: {test_mae:.2f}")

# Compute R-squared on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")

# -------------------------------------------------------------------
# 7. Predict on new data (example)
# -------------------------------------------------------------------
new_data = np.array([[
    7420, 4, 2, 3,
    1, 0, 0, 0, 1,  # mainroad=1, guestroom=0, basement=0, hotwater=0, aircon=1
    2, 1,           # parking=2, prefarea=1
    1, 0, 0         # furnishingstatus_(furnished=1, semi-furnished=0, unfurnished=0)
]])

new_data_scaled = scaler.transform(new_data)
predicted_price = model.predict(new_data_scaled)
print(f"Predicted Price: {predicted_price[0][0]:.2f}")
