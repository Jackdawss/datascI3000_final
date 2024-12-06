import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
csv_file_path = 'Modified_Collisions.csv'
df = pd.read_csv(csv_file_path)

missing_values = df.isnull().sum() / len(df) * 100
columns_to_drop = missing_values[missing_values > 10].index
data = df.drop(columns=columns_to_drop)

missing_cols = data.columns[data.isnull().any()]
for col in missing_cols:
    if pd.api.types.is_numeric_dtype(data[col]):
        non_missing_data = data[col].dropna().values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde.fit(non_missing_data)
        missing_count = data[col].isnull().sum()
        sampled_values = kde.sample(missing_count).flatten()
        data.loc[data[col].isnull(), col] = sampled_values
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Define the feature columns and the target column
non_predictive_columns = [
    'X', 'Y', 'INCKEY', 'COLDETKEY',
    'OBJECTID', 'REPORTNO', 'INCDATE', 'INCDTTM', 'LOCATION',
    'EXCEPTRSNCODE', 'EXCEPTRSNDESC', 'STATUS', 'SDOT_COLDESC',
    'ST_COLDESC', 'INTKEY', 'SEVERITYDESC', 'PEDROWNOTGRNT'
]
df = df.drop(columns=non_predictive_columns, errors='ignore')

categorical_columns = ['WEATHER', 'ROADCOND', 'LIGHTCOND']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

df = df.dropna()

label_encoder = LabelEncoder()
df['ADDRTYPE'] = label_encoder.fit_transform(df['ADDRTYPE'])
df['COLLISIONTYPE'] = label_encoder.fit_transform(df['COLLISIONTYPE'])
df['JUNCTIONTYPE'] = label_encoder.fit_transform(df['JUNCTIONTYPE'])
df['HITPARKEDCAR'] = label_encoder.fit_transform(df['HITPARKEDCAR'])
df['Season'] = label_encoder.fit_transform(df['Season'])



# Split the data into features (X) and target (y)
X = df.drop(columns=["SEVERITYCODE", "FATALITIES", "INJURIES", "SERIOUSINJURIES"])
y = df["SEVERITYCODE"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=10)

# Train the classifier
dt_classifier.fit(X_train, y_train)



# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=True)
plt.show()