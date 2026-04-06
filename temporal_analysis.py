import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_temporal_trends(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter for total drug overdose deaths
    ind_mask = df['Indicator'] == 'Number of Drug Overdose Deaths'
    df_deaths = df[ind_mask].copy()
    
    # Filter out empty Data Values and parse as numeric
    df_deaths.dropna(subset=['Data Value'], inplace=True)
    df_deaths['Data Value'] = df_deaths['Data Value'].astype(str).str.replace(',', '')
    df_deaths['Data Value'] = pd.to_numeric(df_deaths['Data Value'], errors='coerce')
    df_deaths.dropna(subset=['Data Value'], inplace=True)
    
    # We create a datetime based on Year and Month to plot over time
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_deaths['Month_Num'] = df_deaths['Month'].map(month_map)
    df_deaths = df_deaths.dropna(subset=['Month_Num'])
    
    # Create Date
    df_deaths['Date'] = pd.to_datetime(df_deaths['Year'].astype(str) + '-' + df_deaths['Month_Num'].astype(int).astype(str) + '-01')
    
    # Aggregate data nationwide over time
    national_deaths = df_deaths.groupby('Date')['Data Value'].sum().reset_index()
    national_deaths = national_deaths.sort_values('Date')
    
    # Plot Nationwide Overdose Deaths Trend
    plt.figure(figsize=(12, 6))
    plt.plot(national_deaths['Date'], national_deaths['Data Value'], marker='o', linestyle='-', color='b')
    plt.title('Nationwide Trend of Drug Overdose Deaths (Rolling 12-Month Periods)')
    plt.xlabel('Date')
    plt.ylabel('Number of Deaths')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("national_overdose_trend.png")
    print("Saved nationwide trend to 'national_overdose_trend.png'")
    
    # Analyze Top 5 States by total deaths in the dataset
    state_totals = df_deaths.groupby('State Name')['Data Value'].sum().nlargest(5).index.tolist()
    
    plt.figure(figsize=(12, 6))
    for state in state_totals:
        state_data = df_deaths[df_deaths['State Name'] == state]
        state_data = state_data.groupby('Date')['Data Value'].sum().reset_index().sort_values('Date')
        plt.plot(state_data['Date'], state_data['Data Value'], label=state, marker='x', linestyle='--')
        
    plt.title('Top 5 States - Trend of Drug Overdose Deaths')
    plt.xlabel('Date')
    plt.ylabel('Number of Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("top_states_overdose_trend.png")
    print("Saved state trends to 'top_states_overdose_trend.png'")


if __name__ == "__main__":
    cdc_data_path = "data/VSRR_Provisional_Drug_Overdose_Death_Counts_20260404.csv"
    if os.path.exists(cdc_data_path):
        analyze_temporal_trends(cdc_data_path)
    else:
        print(f"File not found: {cdc_data_path}")
