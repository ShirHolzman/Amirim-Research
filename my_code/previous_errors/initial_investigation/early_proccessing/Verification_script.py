import pandas as pd
import numpy as np
import glob
import os

base_path = r'C:\Users\shirh\OneDrive - huji.ac.il\Amirim Research\my_code\EDA_set'
eda_schedules = ['schedule_4', 'schedule_5', 'schedule_7']

all_eda_data = [] # To store the results
excluded_files_log = []
rt_verification_samples = []

print("Running Data Verification Audit & Processing...\n")

for sched in eda_schedules:
    folder_path = os.path.join(base_path, sched, '*.csv')
    files = glob.glob(folder_path)
    
    for file in files:
        filename = os.path.basename(file)
        
        if 'invalid_bias' in filename.lower():
            excluded_files_log.append({'Schedule': sched, 'File': filename, 'Reason': 'Pre-marked as invalid_bias'})
            continue
            
        try:
            df = pd.read_csv(file)
            
            if 'RT' in df.columns and 'side_choice' in df.columns:
                choice_counts = df['side_choice'].value_counts()
                
                if len(choice_counts) < 2 or choice_counts.min() < 5:
                    excluded_files_log.append({'Schedule': sched, 'File': filename, 'Reason': 'Chose one side < 5 times'})
                    continue
                
                # Calculations
                df['RT_Original'] = df['RT']
                df['RT_Clipped'] = np.clip(df['RT'], a_min=1500, a_max=None)
                df['RT_Net'] = df['RT_Clipped'] - 1500
                
                # Z-Score (Normalization per participant)
                rt_std = df['RT_Net'].std()
                df['RT_zscore'] = (df['RT_Net'] - df['RT_Net'].mean()) / rt_std if rt_std > 0 else 0.0

                # Metadata
                df['subject_file'] = filename
                df['schedule'] = sched
                all_eda_data.append(df)
                
                # --- FIX: Added [0] to .iloc ---
                normal_sample = df[df['RT_Original'] > 1500].head(1)
                if not normal_sample.empty:
                    row = normal_sample.iloc[0] # Corrected here
                    rt_verification_samples.append({'File': filename, 'Trial': row['trial_number'], 
                                                    'Original_RT': row['RT_Original'], 
                                                    'Clipped_RT': row['RT_Clipped'], 
                                                    'Net_RT': row['RT_Net']})
                
                anomaly_sample = df[df['RT_Original'] < 1500].head(1)
                if not anomaly_sample.empty:
                    row = anomaly_sample.iloc[0] # Corrected here
                    rt_verification_samples.append({'File': filename, 'Trial': row['trial_number'], 
                                                    'Original_RT': row['RT_Original'], 
                                                    'Clipped_RT': row['RT_Clipped'], 
                                                    'Net_RT': row['RT_Net']})
                    
        except Exception as e:
            print(f"Error in {filename}: {e}")

# Save the final dataset
if all_eda_data:
    final_df = pd.concat(all_eda_data, ignore_index=True)
    final_df.to_csv('cleaned_eda_master.csv', index=False)
    print(f"\nSUCCESS: Saved {len(final_df)} rows to 'cleaned_eda_master.csv'")

# Logs and Verification Output
print("-" * 50)
print(f"EXCLUDED FILES VERIFICATION LOG (Total: {len(excluded_files_log)})")
print("-" * 50)
if excluded_files_log:
    print(pd.DataFrame(excluded_files_log).to_string(index=False))

print("\n" + "-" * 50)
print("RT CALCULATION VERIFICATION SAMPLES")
print("-" * 50)
if rt_verification_samples:
    rt_df = pd.DataFrame(rt_verification_samples).drop_duplicates(subset=['File', 'Original_RT'])
    print(rt_df.head(10).to_string(index=False))

