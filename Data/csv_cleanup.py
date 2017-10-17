import pandas as pd

df = pd.read_csv('na_data_backup.csv')


searchfor = ['SUMM_STD']
df = df[~df.datafmt.str.contains('|'.join(searchfor))]

df.reset_index(drop=True, inplace=True)

df.to_csv('na_data_edit.csv', index=None)