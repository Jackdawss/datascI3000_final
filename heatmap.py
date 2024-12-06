import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
csv_file_path = 'Collisions.csv'
df = pd.read_csv(csv_file_path)

# Ensure the date/time column is in datetime format
df['INCDATE'] = pd.to_datetime(df['INCDATE'])

# Extract the month from the crash date/time
df['Month'] = df['INCDATE'].dt.month

# Map months to seasons
season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
df['Season'] = df['Month'].map(season_mapping)

# Aggregate the data to count crashes by season
crash_counts = df['Season'].value_counts().reindex(['Winter', 'Spring', 'Summer', 'Fall']).fillna(0)

# Create a heatmap
plt.figure(figsize=(10, 1))
sns.heatmap(crash_counts.values.reshape(1, -1), cmap='coolwarm', annot=True, fmt='g', cbar=False)
plt.title('Heatmap of Car Crashes by Season')
plt.xlabel('Season')
plt.yticks([])
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['Winter', 'Spring', 'Summer', 'Fall'], rotation=0)
plt.show()