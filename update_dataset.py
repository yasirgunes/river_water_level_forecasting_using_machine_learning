#!/usr/bin/env python3
"""
Update River Water Level Dataset with Latest Data
=================================================
This script fetches the latest USGS water level data and weather data
up to the current date, ensuring your dataset is always up-to-date
for real-time forecasting.
"""

import hydrofunctions as hf
import pandas as pd
from datetime import datetime, date
from meteostat import Point, Daily, Hourly
import os

print("="*60)
print("UPDATING DATASET WITH LATEST DATA")
print("="*60)

# Get today's date
today = date.today()
print(f"Today's date: {today}")

# Check existing dataset to find where to start
if os.path.exists("combined_dataset.csv"):
    existing_df = pd.read_csv("combined_dataset.csv", parse_dates=['datetime'], index_col='datetime')
    last_date = existing_df.index.max()
    print(f"Existing dataset ends at: {last_date}")
    
    # Start from the day after the last available date
    start_date = (last_date + pd.Timedelta(days=1)).date()
    print(f"Will fetch new data from: {start_date}")
else:
    # If no existing dataset, start from scratch
    start_date = datetime(1995, 10, 1).date()
    print("No existing dataset found. Starting from 1995-10-01")

# Only proceed if there's new data to fetch
if start_date > today:
    print("\n✅ Dataset is already up-to-date!")
    exit(0)

print(f"\n📊 Fetching data from {start_date} to {today}...")

# --- 1. FETCH WATER LEVEL DATA ---
print("\n--- Fetching USGS Water Level Data ---")
site = "03443000"

# Convert dates to strings for hydrofunctions
start_str = start_date.strftime("%Y-%m-%d")
end_str = today.strftime("%Y-%m-%d")

# Get both stage and discharge data
try:
    nwis_data = hf.NWIS(site, 'dv', start_str, end_str, parameterCd=['00065', '00060'])
    df = nwis_data.df()
    
    # Extract both stage and discharge series
    stage_col = f'USGS:{site}:00065:00003'
    discharge_col = f'USGS:{site}:00060:00003'
    
    stage = df[stage_col].rename('stage_ft')
    discharge = df[discharge_col].rename('discharge_cfs')
    
    # Combine both series
    water_data = pd.DataFrame({
        'stage_ft': stage,
        'discharge_cfs': discharge
    })
    
    # Handle gaps
    is_missing = water_data.isnull().any(axis=1)
    gap_groups = is_missing.groupby((~is_missing).cumsum())
    long_gap_indices = gap_groups.filter(lambda g: g.sum() > 3).index
    
    if not long_gap_indices.empty:
        print(f"Removing {len(long_gap_indices)} rows due to gaps > 3 days.")
        water_data = water_data.drop(index=long_gap_indices)
    
    # Interpolate short gaps
    water_data.interpolate(method='linear', inplace=True)
    
    # Convert units
    water_data['stage_m'] = (water_data['stage_ft'] * 0.3048).round(3)
    water_data['discharge_cms'] = (water_data['discharge_cfs'] * 0.0283168).round(3)
    
    # Clean up
    water_df_new = water_data[['stage_m', 'discharge_cms']]
    water_df_new.index.name = 'datetime'
    
    # Remove timezone info to match weather data
    water_df_new.index = water_df_new.index.tz_localize(None)
    
    print(f"✓ Fetched {len(water_df_new)} days of water level data")
    
except Exception as e:
    print(f"❌ Error fetching water data: {e}")
    water_df_new = pd.DataFrame()

# --- 2. FETCH WEATHER DATA ---
print("\n--- Fetching Weather Data ---")
location = Point(35.4333, -82.0333, 660)

try:
    # Fetch daily weather data
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(today, datetime.min.time())
    
    df_met = Daily(location, start_datetime, end_datetime).fetch()
    
    # Select columns
    df_met = df_met[['tavg', 'prcp', 'wspd', 'pres']].round(2)
    df_met.index.name = 'datetime'
    
    # Add relative humidity
    hourly_met = Hourly(location, start_datetime, end_datetime).fetch()
    if not hourly_met.empty:
        df_met['rhum'] = hourly_met['rhum'].resample('D').mean().round(2)
    
    # Interpolate missing values
    df_met = df_met.interpolate(method='time')
    
    print(f"✓ Fetched {len(df_met)} days of weather data")
    
except Exception as e:
    print(f"❌ Error fetching weather data: {e}")
    df_met = pd.DataFrame()

# --- 3. COMBINE NEW DATA ---
if not water_df_new.empty and not df_met.empty:
    new_combined = df_met.join(water_df_new, how='inner')
    new_combined.dropna(inplace=True)
    
    # Reorder columns
    first_cols = ['stage_m', 'discharge_cms']
    remaining_cols = [col for col in new_combined.columns if col not in first_cols]
    new_combined = new_combined[first_cols + remaining_cols]
    
    print(f"\n✓ Combined new data: {len(new_combined)} complete records")
    
    # --- 4. MERGE WITH EXISTING DATA ---
    if os.path.exists("combined_dataset.csv"):
        # Load existing data
        existing_df = pd.read_csv("combined_dataset.csv", parse_dates=['datetime'], index_col='datetime')
        
        # Combine old and new data
        updated_df = pd.concat([existing_df, new_combined])
        
        # Remove any duplicates (keeping the latest)
        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
        
        # Sort by date
        updated_df.sort_index(inplace=True)
        
        print(f"\n📊 Dataset Update Summary:")
        print(f"  - Previous records: {len(existing_df)}")
        print(f"  - New records added: {len(new_combined)}")
        print(f"  - Total records now: {len(updated_df)}")
        print(f"  - Date range: {updated_df.index.min()} to {updated_df.index.max()}")
        
    else:
        updated_df = new_combined
        print(f"\n📊 New dataset created with {len(updated_df)} records")
    
    # --- 5. SAVE UPDATED DATASET ---
    # Backup existing file
    if os.path.exists("combined_dataset.csv"):
        backup_name = f"combined_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.rename("combined_dataset.csv", backup_name)
        print(f"\n💾 Backed up existing dataset to: {backup_name}")
    
    # Save updated dataset
    updated_df.to_csv("combined_dataset.csv")
    print(f"✅ Updated dataset saved to: combined_dataset.csv")
    
    # Display sample of new data
    if len(new_combined) > 0:
        print(f"\n🔍 Sample of newly added data:")
        print(new_combined.tail(5))
    
else:
    print("\n❌ Could not fetch new data. Please check your internet connection.")

print("\n" + "="*60)
print("UPDATE COMPLETE!")
print("="*60) 