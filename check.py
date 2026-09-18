import pandas as pd
df = pd.read_csv(r"data_raw\atp_tennis.csv")  # actual filename
print(df.shape)
print(df['Date'].min(), df['Date'].max())
print(df['Surface'].value_counts())
print(df['Tournament'].str.contains('Wimbledon').sum(), "Wimbledon matches")