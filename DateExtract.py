import pandas as pd

csv_file_path = 'Collisions.csv'
new_csv_file_path = 'Modified_Collisions.csv'

df = pd.read_csv(csv_file_path)

df = df[df['SEVERITYCODE'] != 0]

df['SEVERITYCODE'] = df['SEVERITYCODE'].replace({'2b': 3})

df['UNDERINFL'] = df['UNDERINFL'].replace({'N': 0, 'Y': 1})
df['INATTENTIONIND'] = df['INATTENTIONIND'].replace({'N': 0, 'Y': 1})
df['SPEEDING'] = df['SPEEDING'].replace({'N': 0, 'Y': 1})
df['UNDERINFL'] = pd.to_numeric(df['UNDERINFL'], errors='coerce').fillna(0).astype(int)
df['INATTENTIONIND'] = pd.to_numeric(df['INATTENTIONIND'], errors='coerce').fillna(0).astype(int)
df['SPEEDING'] = pd.to_numeric(df['SPEEDING'], errors='coerce').fillna(0).astype(int)
df['High_Risk_Driver'] = df['INATTENTIONIND'] + df['UNDERINFL']

weather_risk = {'Clear': 1, 'Overcast': 2, 'Raining': 3, 'Fog/Smog': 4, 'Other': 2, 'Unknown':1}
roadcond_risk = {'Dry': 1, 'Wet': 2, 'Standing Water': 3, 'Snow/Slush': 4, 'Ice': 5, 'Other': 2, 'Unknown': 1}
visibility_risk = {'Daylight': 1, 'Dawn': 2, 'Dusk': 3, 'Dark - Street Lights On': 4, 'Dark - No Street Lights': 5, 'Unknown': 1}

df['Environmental_Risk'] = df['WEATHER'].map(weather_risk) + df['LIGHTCOND'].map(visibility_risk) + df['ROADCOND'].map(roadcond_risk)

# Extract month from INCDATE and map to season
df['INCDATE'] = pd.to_datetime(df['INCDATE'])
df['Month'] = df['INCDATE'].dt.month

season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}

df['Season'] = df['Month'].map(season_mapping)

df['Weekday'] = df['INCDATE'].dt.dayofweek

df.to_csv(new_csv_file_path, index=False)