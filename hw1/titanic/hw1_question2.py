import pandas as pd
def startup():	
	data = pd.read_csv('train.csv')
	return data

all_data=startup()
any_nans=all_data[all_data.isnull().any(axis=1)]
print(any_nans.shape)
print(all_data.shape)

