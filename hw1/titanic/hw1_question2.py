import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# df.info()
# df.describe()

def startup():
	data = pd.read_csv('train.csv')
	return data

def test_startup():
	data =pd.read_csv('test.csv')
	return data

def numberify_sex(sex):
	if sex=="female":
		return 0
	else:
		return 1

def sex_to_num(all_data): 
	all_data['CleanSex'] = all_data['Sex'].map( numberify_sex ).astype(int)
	return all_data

def numberify_embark(val):
	if val=="C":
		return 0
	elif val=="Q":
		return 1
	else:
		return 2

def embark_to_num(all_data):
	all_data['CleanEmbarked'] = all_data['Embarked'].map( numberify_embark ).astype(int)
	return all_data

def fill_age(all_data):
	all_data['CleanAge'] = all_data['Age']
	all_data["CleanAge"]=all_data["CleanAge"].fillna(int(all_data["CleanAge"].mean()))
	return all_data

def fill_fare(all_data):
	all_data['CleanFare'] = all_data['Fare']
	all_data["CleanFare"]=all_data["CleanFare"].fillna(int(all_data["CleanFare"].mean()))
	return all_data


def clean_data(all_data):
	all_data=sex_to_num(all_data)
	all_data=embark_to_num(all_data)
	all_data=fill_age(all_data)
	all_data=fill_fare(all_data)
	all_data=all_data.drop(["Sex","Embarked","Age","Cabin","Name","Ticket","PassengerId","Fare"],axis=1)
	return all_data

def create_model():
	# http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
	all_data=startup()
	cleaned_data=clean_data(all_data)

	survived=cleaned_data["Survived"]
	#Although it should really be called boating data hahaha... 
	training=cleaned_data.drop(["Survived"],axis=1) 

	lr = LogisticRegression()
	lr.fit(training.values,survived.values)	

	return (lr,training,survived)

def self_test():
	model,training,survived=create_model()

	predicted= model.predict(training)

	print(metrics.accuracy_score(survived, predicted))

def actual_test():
	model,training,survived=create_model()

	test_data=test_startup()
	cleaned_test=clean_data(test_data)
	# print (cleaned_test.info())

	predicted= model.predict(cleaned_test)
	np.savetxt("titanic.csv",predicted,delimiter=",")


#self_test()
actual_test()



