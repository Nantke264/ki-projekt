import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data=pd.read_csv('Housing.csv')

encoder = LabelEncoder()

encoding = ['furnishingstatus', 'prefarea', 'airconditioning',
            'hotwaterheating', 'basement', 'guestroom', 'mainroad']

for column in encoding:
    data[column] = encoder.fit_transform(data[column])

plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=1, cbar=True)
plt.show()