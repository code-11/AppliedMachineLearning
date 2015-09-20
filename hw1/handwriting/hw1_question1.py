import csv 
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib
from progressbar import ProgressBar
import numpy as np
import scipy.spatial.distance as sp
import scipy
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import itertools
import math
import cPickle as pickle

# return list of tuples [(label0, [pixel0, pixel1],...,pixel784), ...]
def setup():
	data_file=open("train.csv","rb")
	data=csv.reader(data_file,delimiter=",")
	digits=[]
	index=0
	for row in data:
		if index!=0:
			digits.append((row[0],row[1:]))
		index+=1
	return digits

# slightly optimized version of setup
def quick_setup():
	data = pd.read_csv('train.csv')
	return data

# loads test data
def quick_test_setup():
	data =pd.read_csv('test.csv')
	return data

# prints lengths of all_data and all_data duplicates removed
def duplicate_check(all_data):
	labels,data=modify_structure(all_data)
	data_set=set()
	for el in data:
		data_set.add(tuple(el))
	print(len(data))
	print(len(data_set))

# displays image of pixel array
def make_image(raw_data,num):
	img = Image.new( 'RGB', (28,28), "black") # create a new black image
	pixels = img.load() # create the pixel map
	index=0
	for i in range(img.size[0]):    # for every pixel:
	    for j in range(img.size[1]):
	    	val= int(raw_data[index])
        	pixels[j,i] = (val,val,val)
	        index+=1
	img.save(str(num)+".jpg", "JPEG")

# finds first example of a unique digit 
# returns data without the example digits
def find_each(all_data):
	seen_data=[]
	seen_label=[]
	new_all_data=[]
	for i in range(len(all_data)):
		el = all_data[i]
		if not(el[0] in seen_label):
			# print(el[0], seen_label)
			seen_label.append(el[0])
			seen_data.append(el[1])
		else:
			# print('dont add me')
			new_all_data.append(el)
	return (new_all_data,seen_label, seen_data)

# displays each unique digit
def display_each(all_data):
	new_all_data,seen_label, seen_data = find_each(all_data)
	for index in range(len(seen_label)):
		print(seen_label[index])
		make_image(seen_data[index],index)

# converts [(), (), ...] to ([], [], ...)
def modify_structure(all_data):
	no_dups_data = []
	no_dups_label = []
	for i in all_data:
			no_dups_label.append(i[0])
			no_dups_data.append(i[1])
	return (np.array(no_dups_label).astype(int),np.array(no_dups_data).astype(int))

# finds kth nearest neightbor of the unique digits where k=1
# prints the digit label and the nearest neighbor guess
def firstKNN(all_data):
	data_no_digits, label, digits = find_each(all_data)
	digits = scipy.array(digits).astype(int)
	no_dig_label,no_dig_data = modify_structure(data_no_digits)
	for k in range(len(digits)):
		distance = []
		for i in range(len(no_dig_data)):
			distance.append(sp.euclidean(no_dig_data[i],digits[k]))
		mindex = np.argmin(np.array(distance))
		if(int(no_dig_label[mindex]) == int(label[k])):
			print "actual",label[k], ", label of nn", no_dig_label[mindex] #, distance[mindex]
		else:
			print "actual",label[k], ", label of nn", no_dig_label[mindex], "*"

# calculates the distance for knn
# returns tuple of label and distance
def calc_distances_knn(row,digit):
	np_row=row
	label=np_row[0]
	pixl=np.delete(np_row,0)
	dis=sp.euclidean(pixl,digit)
	return (label,dis)

# calculates the majority vote for knn
# returns which label is most popular 
def majority_vote(all_data,nn_indexes):
	label_counts={}
	for index in nn_indexes:
		label=all_data.loc[index]["label"]
		if label in label_counts:
			label_counts[label]+=1
		else:
			label_counts[label]=1
	return max(label_counts, key=label_counts.get)

# our implementation of k nearest neightbor 
def kNN(all_data, digit, k):
	labels=np.array([])
	dists=np.array([])
	
	pbar=ProgressBar()
	for row in all_data.values:
		label,dist=calc_distances_knn(row,digit)
		labels=np.append(labels,label)
		dists=np.append(dists,dist)

	sorted_indicies=np.argsort(dists)
	sorted_pandas_indices=all_data.index.values[[sorted_indicies]]
	nearest_neighbor_indexes=sorted_pandas_indices[:k]
	return majority_vote(all_data,nearest_neighbor_indexes)

# preforms knn on many digits
# returns label perdictions
def KNNBatch(all_data, digits, k):
	results=np.array([])
	pbar=ProgressBar()

	for digit in pbar(digits):
		results=np.append(results,kNN(all_data,digit,k))
	return results

# calculated the three fold cross vaidation 
# returns tuple of all tests and results
def three_cross_validation(all_data):
	only_pixls= all_data.drop("label",axis=1)
	num_rows=all_data.shape[0]
	kf = cross_validation.KFold(num_rows, shuffle=True, n_folds=3)

	all_results=np.array([])
	all_tests=np.array([])

	print("[Classifying]")
	for train_indexes, test_indexes in kf:
		training_df = all_data.loc[train_indexes]
		testing_data = only_pixls.loc[test_indexes].values

		all_labels=all_data["label"]
		testing_labels=all_labels.loc[test_indexes].values

		#K value of 5 chosen arbitrarily
		result_labels=KNNBatch(training_df, testing_data,5) 
		
		all_results=np.append(all_results,result_labels)
		all_tests=np.append(all_tests,testing_labels)

	return (all_tests,all_results)

# produces the accuracy of our knn model
# returns float of accuracy
def right_or_wrong(all_results,all_tests):
	right=0
	wrong=0
	for i in xrange(len(all_results)):
		if all_results[i]==all_tests[i]:
			right+=1
		else:
			wrong+=1
	return right/float(right+wrong)

# calculates the prior of a digit in the data
# saves results to a pdf
def get_probs(all_data):
	labels={}
	prior_probs={}
	for el in all_data:
		if not( el[0] in labels):
			labels[el[0]]=1
		else:
			labels[el[0]]+=1
	norm=float(sum(labels.values()))

	for key in labels.keys():
		prior_probs[key]=labels[key]/norm
		
	ind = np.arange(len(prior_probs.keys()))
	plt.bar(ind,prior_probs.values())
	plt.savefig("prior_probs.pdf")

# gets data by label 
# returns data with label
def get_data_by_label(all_data, label):
	data_with_label = []
	for data in all_data:
		if(int(data[0]) == int(label) ):
			data_with_label.append(data)
	return data_with_label 	

# converst all data into scipy arrays
# returns scipy array
def convert_scipy(all_data):
	new_data = []
	for label, data in all_data:
		sci_data = scipy.array(data).astype(int)
		new_data.append((label, sci_data))
	return new_data

# calculates the combination of ones and zeros
# finds the diferences between pair 
# returns genuine and imposter distance vectors
def  binary_distances(all_data):
	zeros = get_data_by_label(all_data, 0)
	ones = get_data_by_label(all_data, 1)

	#Drop some of the data because the combinations are too large for us
	#canthandletheData
	zeros=zeros[50:250]
	ones=ones[50:250]

	sci_zeros = convert_scipy(zeros)
	sci_ones = convert_scipy(ones)
	ones_and_zeros=sci_ones+sci_zeros

	combinations = itertools.combinations(ones_and_zeros, 2)
	imposter = np.array([])
	genuine = np.array([])
	for el1, el2 in combinations:
		dist = sp.euclidean(el1[1], el2[1])
		if (el1[0] == el2[0]):
			genuine = np.append(genuine,dist) 
		else:
			imposter = np.append(imposter,dist) 
	print "average imposter: ", np.mean(imposter), " average genuine: ", np.mean(genuine)
	return (genuine, imposter)

#Returns a tuble with True if genuine and the distance between the digits
def dists_from_combos(data,tup):
	lbl1 = data.loc[tup[0]]["label"]
	lbl2 = data.loc[tup[1]]["label"]

	#may be accidently including label in the distance calculation
	pixl1 = data.loc[tup[0]].values
	pixl2 = data.loc[tup[1]].values

	return (lbl1==lbl2,sp.euclidean(pixl1,pixl2))

# an optimized version which calculated the distances between
# all combinations of ones and zeros
# returns tuple of geniune and imposter vectors
def quick_binary_distances(data):
	print("[Load Data]")
	both = data.loc[ (data["label"] == 1) | (data["label"] == 0)]

	print("[Find Indicies]")
	indices = both.index.values

	print("[Taking Combinations]")
	all_combinations = itertools.combinations(indices, 2)

	print("[Converting to np]")
	all_c = np.array(list(all_combinations))
 
 	print("[Calculate Distances from Indicies]")

 	genuine=np.array([])
 	imposter=np.array([])
 	pbar=ProgressBar()

 	for i in pbar(xrange(len(all_c))):

 		gen, dis= dists_from_combos(data,all_c[i])
 		if (gen):
 			genuine=np.append(genuine,dis)
 		else:
 			imposter=np.append(imposter,dis)

 	return (genuine, imposter)


# plots histogram of distances
# returns plot
def plot_distances(data_list):
	bins, edges = np.histogram(data_list, 150, normed=1)
	left,right = edges[:-1],edges[1:]
	X = np.array([left,right]).T.flatten()
	Y = np.array([bins,bins]).T.flatten()
	return plt.plot(X,Y)

# calculates the true positive rate and true negative rate
# based on given threshold and returns them in a tuple
def roc_plot_helper(genuine, imposter, thresh_dist):
	TPR = sum(genuine[np.where(genuine<thresh_dist)]) / float(sum(genuine))
	TNR = sum(imposter[np.where(imposter<thresh_dist)]) / float(sum(imposter))
	return (TPR, TNR)

# plots the roc curve and returns the plot
def roc_plot(genuine, imposter):
	TPR = []
	TNR = []
	m_gen = max(genuine)
	m_imp = max(imposter) 
	for thresh in range(int(max([m_gen, m_imp]))):
		tpr, tnr = roc_plot_helper(genuine, imposter, thresh)
		TPR.append(tpr)
		TNR.append(tnr)
	return plt.plot(TNR, TPR)

# http://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

# run me for question 1 part b
def partb():
	all_data = setup()
	display_each(all_data)

# run me for question 1 part c
def partc():
	all_data = setup()
	get_probs(all_data)

# knn where k = 1
def partd():
	all_data = setup()
	firstKNN(all_data)

# calculates distances on MNIST dataset for 1's and 0's
def parte():
	all_data = quick_setup().head(10000)
	genuine,imposter = quick_binary_distances(all_data)
	genuine_plot = plot_distances(genuine)
	imposter_plot = plot_distances(imposter)
	genuine_patch = matplotlib.patches.Patch(color = 'blue', label = 'genuine')
	imposter_patch = matplotlib.patches.Patch(color = 'green', label = 'imposter')
	plt.legend(handles = [genuine_patch, imposter_patch])
	plt.title("Binary Distances")
	plt.savefig("parte.png")
	plt.show()


# run this for our original implementation for part e
def parte2():
	print("start")
	all_data = setup()
	print("data setup")
	genuine,imposter=binary_distances(all_data)
	genuine_plot = plot_distances(genuine)
	imposter_plot = plot_distances(imposter)
	genuine_patch = matplotlib.patches.Patch(color = 'blue', label = 'genuine')
	imposter_patch = matplotlib.patches.Patch(color = 'green', label = 'imposter')
	plt.legend(handles = [genuine_patch, imposter_patch])
	plt.show()


def partef():
	all_data = quick_setup().head(10000)
	genuine,imposter = quick_binary_distances(all_data)

	#parte
	genuine_plot = plot_distances(genuine)
	imposter_plot = plot_distances(imposter)
	genuine_patch = matplotlib.patches.Patch(color = 'blue', label = 'genuine')
	imposter_patch = matplotlib.patches.Patch(color = 'green', label = 'imposter')
	plt.legend(handles = [genuine_patch, imposter_patch])
	plt.title("Binary Distances")
	plt.savefig("parte.png")

	plt.clf()
	plt.close()

	#partf
	roc_plot(genuine,imposter)
	plt.title("ROC Curve")
	plt.xlabel("TNR")
	plt.ylabel("TPR")
	plt.savefig("partf.png")

# plot the roc curve on all points using parte
def partf():
	all_data = quick_setup()
	genuine, imposter = quick_binary_distances(all_data)
	roc_plot(genuine,imposter)
	plt.show()

# plots an roc curve using parte2
def partf2():
	all_data = setup()
	genuine, imposter = binary_distances(all_data)
	plot = roc_plot(genuine, imposter)
	plt.set_xlabel("TNR")
	plt.set_ylabel("TPR")
	plt.show()

# knn classifier
def partg():
	all_data = quick_setup()
	kNN(all_data,np.zeros(784),5)

# 3 fold cross validation
def parth():
	print("[Loading Data]")
	all_data = quick_setup().head(4000)
	all_tests,all_results=three_cross_validation(all_data)
	print(right_or_wrong(all_tests,all_results))

# confusion matrix
def parti():
	print("[Loading Data]")
	all_data = quick_setup().head(500)
	all_tests,all_results=three_cross_validation(all_data)
	actu = pd.Series(all_tests, name='Actual')
	pred = pd.Series(all_results, name='Predicted')
	df_confusion = pd.crosstab(actu, pred)
	plot_confusion_matrix(df_confusion)
	plt.savefig("parti.png")
	# print(confusion_matrix(all_results,all_tests,[0,1,2,3,4,5,6,7,8,9]))


def parthi():
	print("[Loading Data]")
	all_data = quick_setup().head(5000)
	all_tests,all_results=three_cross_validation(all_data)

	#part h
	f = open('parth.txt', 'w')
	f.write(str(right_or_wrong(all_tests,all_results)))

	#part i
	actu = pd.Series(all_tests, name='Actual')
	pred = pd.Series(all_results, name='Predicted')
	df_confusion = pd.crosstab(actu, pred)
	plot_confusion_matrix(df_confusion)
	plt.savefig("parti.png")

def partj():
	all_data =quick_setup().head(5000)
	all_test =quick_test_setup().values
	predictions=KNNBatch(all_data, all_test, 5)
	np.savetxt("partj.csv",predictions,delimiter=",")



# partb()
# partc()
# partd()
# parte()
# parte2()
# partf()
# partf2()
# partef()
# parth()
# parti()
# parthi()
# partj()


