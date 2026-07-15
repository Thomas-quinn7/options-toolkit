import requests
import pandas as pd
from datetime import datetime

def get_historical_weather(lat: float, lon: float, start_date: str, end_date: str):
    """Fetch historical weather from Open-Meteo (no API key needed)"""
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': [
            'temperature_2m_max',
            'temperature_2m_min', 
            'temperature_2m_mean',
            'precipitation_sum',
            'windspeed_10m_max',
            'sunshine_duration'
        ],
        'timezone': 'auto'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'temp_mean': data['daily']['temperature_2m_mean'],
            'precipitation': data['daily']['precipitation_sum'],
            'wind_speed': data['daily']['windspeed_10m_max'],
            'sunshine_duration': data['daily']['sunshine_duration']
        })
        
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

if __name__ == "__main__":
    # London coordinates
    london_lat, london_lon = 51.5074, -0.1278
    
    print("Fetching historical weather data for London...")
    
    # Get full year of data
    df = get_historical_weather(
        london_lat, 
        london_lon, 
        "2022-01-01", 
        "2024-12-31"
    )
    
    if df is not None:
        print("\n✓ Successfully fetched data!")
        print(f"\nFirst 10 days:")
        print(df.head(10))
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        print(f"\nBasic statistics:")
        print(df.describe())
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Save to CSV
        df.to_csv('london_weather_2023.csv')
        print("\n✓ Data saved to 'london_weather_2023.csv'")
    else:
        print("Failed to fetch data")