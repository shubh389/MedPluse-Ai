import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_hospital_data():
    """Generate synthetic hospital admission data for multiple cities"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
    start_date = '2024-01-01'
    end_date = '2025-10-10'  # Current date
    
    # Festival dates for 2024-2025
    festivals = {
        '2024-01-01': 'New Year',
        '2024-03-08': 'Holi',
        '2024-03-29': 'Good Friday',
        '2024-04-11': 'Eid',
        '2024-08-15': 'Independence Day',
        '2024-08-26': 'Janmashtami',
        '2024-10-02': 'Gandhi Jayanti',
        '2024-10-12': 'Dussehra',
        '2024-11-01': 'Diwali',
        '2024-11-15': 'Guru Nanak Jayanti',
        '2024-12-25': 'Christmas',
        '2025-01-01': 'New Year',
        '2025-03-14': 'Holi',
        '2025-03-30': 'Eid',
        '2025-08-15': 'Independence Day',
        '2025-10-02': 'Gandhi Jayanti',
        '2025-10-20': 'Dussehra',
        '2025-11-01': 'Diwali',
        '2025-12-25': 'Christmas'
    }
    
    # City-specific configurations
    city_configs = {
        'Delhi': {
            'base_patients': 150,
            'aqi_mean': 200,
            'aqi_std': 80,
            'temp_mean': 24,
            'temp_std': 8,
            'seasonal_amplitude': 20,
            'outbreak_probability': 0.05
        },
        'Mumbai': {
            'base_patients': 180,
            'aqi_mean': 120,
            'aqi_std': 50,
            'temp_mean': 28,
            'temp_std': 4,
            'seasonal_amplitude': 15,
            'outbreak_probability': 0.04
        },
        'Bangalore': {
            'base_patients': 120,
            'aqi_mean': 80,
            'aqi_std': 40,
            'temp_mean': 22,
            'temp_std': 3,
            'seasonal_amplitude': 10,
            'outbreak_probability': 0.03
        },
        'Chennai': {
            'base_patients': 140,
            'aqi_mean': 100,
            'aqi_std': 45,
            'temp_mean': 30,
            'temp_std': 3,
            'seasonal_amplitude': 12,
            'outbreak_probability': 0.04
        },
        'Kolkata': {
            'base_patients': 130,
            'aqi_mean': 150,
            'aqi_std': 60,
            'temp_mean': 26,
            'temp_std': 6,
            'seasonal_amplitude': 18,
            'outbreak_probability': 0.04
        }
    }
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    all_data = []
    
    for city in cities:
        config = city_configs[city]
        city_outbreak_state = 0
        outbreak_duration = 0
        
        print(f"Generating data for {city}...")
        
        for date in dates:
            # Base patient load with seasonal variation
            day_of_year = date.dayofyear
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365.25)
            base_patients = config['base_patients'] + config['seasonal_amplitude'] * seasonal_factor
            
            # Weekly pattern (higher on weekdays, lower on weekends)
            day_of_week = date.weekday()
            if day_of_week >= 5:  # Weekend
                weekly_factor = 0.85
            else:  # Weekday
                weekly_factor = 1.1
            
            # Generate environmental factors
            # AQI with seasonal and random variation
            if date.month in [11, 12, 1, 2]:  # Winter months, higher AQI
                aqi_seasonal_boost = 50
            elif date.month in [6, 7, 8, 9]:  # Monsoon, lower AQI
                aqi_seasonal_boost = -30
            else:
                aqi_seasonal_boost = 0
            
            aqi = max(20, np.random.normal(
                config['aqi_mean'] + aqi_seasonal_boost, 
                config['aqi_std']
            ))
            
            # Temperature with seasonal variation
            temp_seasonal = 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Peak in May
            temperature = np.random.normal(
                config['temp_mean'] + temp_seasonal,
                config['temp_std']
            )
            
            # Festival effect
            date_str = date.strftime('%Y-%m-%d')
            festival = festivals.get(date_str, '')
            festival_effect = 0
            
            if festival:
                if festival in ['Diwali', 'Holi', 'New Year']:
                    festival_effect = np.random.uniform(40, 80)  # Major festivals
                else:
                    festival_effect = np.random.uniform(15, 35)  # Minor festivals
            
            # Outbreak simulation
            if city_outbreak_state == 0:  # No outbreak
                if np.random.random() < config['outbreak_probability']:
                    city_outbreak_state = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                    outbreak_duration = np.random.randint(7, 30)  # 1-4 weeks
            else:  # Ongoing outbreak
                outbreak_duration -= 1
                if outbreak_duration <= 0:
                    city_outbreak_state = 0
            
            # Calculate outbreak effect on admissions
            outbreak_effects = {0: 0, 1: 15, 2: 35, 3: 60}
            outbreak_effect = outbreak_effects[city_outbreak_state]
            
            # AQI effect on respiratory admissions
            if aqi > 100:
                aqi_effect = (aqi - 100) * 0.3  # 0.3 patients per AQI point above 100
            else:
                aqi_effect = 0
            
            # Temperature effect (extreme temperatures increase admissions)
            if temperature < 10 or temperature > 35:
                temp_effect = abs(temperature - 22.5) * 0.5
            else:
                temp_effect = 0
            
            # Calculate total patients
            total_patients = (
                base_patients * weekly_factor +
                festival_effect +
                outbreak_effect +
                aqi_effect +
                temp_effect +
                np.random.normal(0, 8)  # Random noise
            )
            
            # Ensure minimum and reasonable maximum
            total_patients = max(20, min(500, total_patients))
            
            # Round to integer
            patients_admitted = int(round(total_patients))
            
            all_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'city': city,
                'aqi': int(round(aqi)),
                'temperature': round(temperature, 1),
                'festival': festival,
                'outbreak': city_outbreak_state,
                'patients_admitted': patients_admitted
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add some data quality issues (realistic)
    # Missing values (rare)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.001), replace=False)
    df.loc[missing_indices, 'aqi'] = np.nan
    
    # Outliers (equipment malfunction simulation)
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.002), replace=False)
    df.loc[outlier_indices, 'patients_admitted'] *= np.random.uniform(1.5, 2.5, size=len(outlier_indices))
    df['patients_admitted'] = df['patients_admitted'].round().astype(int)
    
    return df

def add_data_statistics(df):
    """Print dataset statistics"""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Cities: {df['city'].nunique()} ({', '.join(df['city'].unique())})")
    
    print(f"\nPatient Admissions:")
    print(f"  Mean: {df['patients_admitted'].mean():.1f}")
    print(f"  Std: {df['patients_admitted'].std():.1f}")
    print(f"  Min: {df['patients_admitted'].min()}")
    print(f"  Max: {df['patients_admitted'].max()}")
    
    print(f"\nAQI:")
    print(f"  Mean: {df['aqi'].mean():.1f}")
    print(f"  Std: {df['aqi'].std():.1f}")
    print(f"  Min: {df['aqi'].min()}")
    print(f"  Max: {df['aqi'].max()}")
    print(f"  Missing: {df['aqi'].isna().sum()}")
    
    print(f"\nTemperature (°C):")
    print(f"  Mean: {df['temperature'].mean():.1f}")
    print(f"  Std: {df['temperature'].std():.1f}")
    print(f"  Min: {df['temperature'].min()}")
    print(f"  Max: {df['temperature'].max()}")
    
    print(f"\nFestivals:")
    festival_counts = df[df['festival'] != '']['festival'].value_counts()
    print(f"  Total festival days: {len(df[df['festival'] != ''])}")
    print(f"  Unique festivals: {len(festival_counts)}")
    if len(festival_counts) > 0:
        print("  Top festivals:")
        for fest, count in festival_counts.head().items():
            print(f"    {fest}: {count} days")
    
    print(f"\nOutbreak levels:")
    outbreak_counts = df['outbreak'].value_counts().sort_index()
    for level, count in outbreak_counts.items():
        pct = (count / len(df)) * 100
        level_name = {0: 'None', 1: 'Localized', 2: 'Regional', 3: 'Epidemic'}[level]
        print(f"  Level {level} ({level_name}): {count:,} days ({pct:.1f}%)")
    
    print(f"\nBy City (Average daily admissions):")
    city_stats = df.groupby('city')['patients_admitted'].agg(['mean', 'std', 'min', 'max'])
    for city in city_stats.index:
        stats = city_stats.loc[city]
        print(f"  {city}: {stats['mean']:.1f} ± {stats['std']:.1f} "
              f"(range: {stats['min']}-{stats['max']})")

if __name__ == "__main__":
    print("Generating synthetic hospital admission data...")
    
    # Generate the data
    df = generate_synthetic_hospital_data()
    
    # Display statistics
    add_data_statistics(df)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/hospital_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Synthetic data saved to: {output_file}")
    print(f"📊 Total records generated: {len(df):,}")
    
    # Save sample for quick inspection
    sample_file = 'data/sample_data.csv'
    df.head(20).to_csv(sample_file, index=False)
    print(f"📋 Sample data (first 20 rows) saved to: {sample_file}")
    
    # Display sample
    print(f"\nSample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    print(f"\n🎉 Data generation completed successfully!")
    print(f"You can now run: python train_model.py")