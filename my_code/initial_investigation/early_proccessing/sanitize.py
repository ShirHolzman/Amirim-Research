import pandas as pd
import numpy as np
import glob
import os

# Define the base path to your data
base_path = r'C:\Users\shirh\OneDrive - huji.ac.il\Amirim Research\my_code\EDA_set'

# We focus ONLY on the EDA schedules for this part of the research
eda_schedules = ['schedule_4', 'schedule_5', 'schedule_7']

all_eda_data = []
total_files_loaded = 0
excluded_files_count = 0

print("Starting Data Preprocessing & Splitting (Part A)...")

for sched in eda_schedules:
    # Construct path for the specific schedule folder
    folder_path = os.path.join(base_path, sched, '*.csv')
    files = glob.glob(folder_path)
    
    for file in files:
        # Ignore explicit invalid_bias files if they exist in the folder
        if 'invalid_bias' in file.lower():
            excluded_files_count += 1
            continue
            
        try:
            df = pd.read_csv(file)
            
            # Ensure the required columns exist
            if 'RT' in df.columns and 'side_choice' in df.columns:
                
                # Filtering outlier participants (less than 5 choices on either side)
                choice_counts = df['side_choice'].value_counts()
                
                # If the participant chose only one side for all 100 trials, 
                # or chose one side less than 5 times, we exclude them.
                if len(choice_counts) < 2 or choice_counts.min() < 5:
                    excluded_files_count += 1
                    continue
                
                # RT Correction
                # Clip times less than 1500ms to exactly 1500ms
                df['RT'] = np.clip(df['RT'], a_min=1500, a_max=None)
                # Subtract the 1.5 seconds (1500ms) hardware delay
                df['RT_net'] = df['RT'] - 1500
                
                # Z-Scoring (Normalization per participant)
                rt_mean = df['RT_net'].mean()
                rt_std = df['RT_net'].std()
                
                # Protect against division by zero 
                if rt_std > 0:
                    df['RT_zscore'] = (df['RT_net'] - rt_mean) / rt_std
                else:
                    df['RT_zscore'] = 0.0
                    
                # Add metadata columns for tracking
                df['subject_file'] = os.path.basename(file)
                df['schedule'] = sched
                
                # Append the clean participant dataframe to our master list
                all_eda_data.append(df)
                total_files_loaded += 1
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Combine all individual dataframes into one large master dataframe
if len(all_eda_data) > 0:
    eda_df = pd.concat(all_eda_data, ignore_index=True)
    print("-" * 40)
    print("Part A Completed Successfully!")
    print(f"Total valid subjects loaded for EDA: {total_files_loaded}")
    print(f"Total subjects excluded (invalid bias): {excluded_files_count}")
    print(f"Total decision rows ready for analysis: {len(eda_df)}")
    eda_df.to_csv(os.path.join(base_path, 'cleaned_eda_data.csv'), index=False)
else:
    print("No data was loaded. Please check the directory paths.")
