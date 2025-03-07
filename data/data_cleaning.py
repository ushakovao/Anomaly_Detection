import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

data_folder = os.path.dirname(os.path.abspath(__file__))
yahoo_file = os.path.join(data_folder, "GME_yahoo_data.csv")
vix_file = os.path.join(data_folder, "VIX_fred_data.csv")
output_file = os.path.join(data_folder, "GME_Cleaned_Data.csv")

gme_yahoo = pd.read_csv(yahoo_file)
vix_fred = pd.read_csv(vix_file)

gme_yahoo['Date'] = pd.to_datetime(gme_yahoo['Date'])
vix_fred['DATE'] = pd.to_datetime(vix_fred['DATE'])

vix_fred.rename(columns={"DATE": "Date"}, inplace=True)

gme_yahoo.rename(columns={'Close': 'Yahoo_Close', 'High': 'Yahoo_High', 'Low': 'Yahoo_Low', 'Open': 'Yahoo_Open', 'Volume': 'Yahoo_Volume'}, inplace=True)
vix_fred.rename(columns={'VIXCLS': 'VIX_Closing'}, inplace=True)

merged_data = pd.merge(gme_yahoo, vix_fred, on="Date", how="outer")

print("Data cleaning and merging completed successfully!")

merged_data = merged_data.sort_values(by="Date")

date_column = merged_data['Date']
merged_data = merged_data.drop(columns=['Date'])

missing_values = merged_data.isnull().sum()
total_rows = len(merged_data)

if missing_values.sum() > 0:
    print("\n=== Missing Data Found ===\n")

    missing_summary = pd.DataFrame({
        "Missing Count": missing_values[missing_values > 0],
        "Percentage": (missing_values[missing_values > 0] / total_rows) * 100
    })
    print(missing_summary)

    print("\nSample rows with missing data:")
    print(merged_data[merged_data.isnull().any(axis=1)].head(10))

    imputer = SimpleImputer(strategy='mean')
    merged_data[merged_data.columns] = imputer.fit_transform(merged_data[merged_data.columns])

    print("Missing data imputed with the mean value.")
else:
    print("No missing data found.")

merged_data['Date'] = date_column

merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])

print(merged_data.columns)

merged_data.to_csv(output_file, index=False)

print(f"\nData cleaning complete! Cleaned data saved to: {output_file}")
