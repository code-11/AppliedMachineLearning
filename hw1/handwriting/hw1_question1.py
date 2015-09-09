import csv 
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import scipy.spatial.distance as sp
import scipy
import itertools

# import data csv data 
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

# prints lengths of all_data and all_data duplicates removed
def duplicate_check(all_data):
	labels,data=modify_structure(all_data)
	data_set=set()
	for el in data:
		data_set.add(tuple(el))
	print(len(data))
	print(len(data_set))

# displays image of pixel array
def make_image(raw_data):
	img = Image.new( 'RGB', (28,28), "black") # create a new black image
	pixels = img.load() # create the pixel map
	index=0
	for i in range(img.size[0]):    # for every pixel:
	    for j in range(img.size[1]):
	    	val= int(raw_data[index])
        	pixels[j,i] = (val,val,val)
	        index+=1
	img.show()

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
		make_image(seen_data[index])

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

def kNN(all_data, digit, k):
	pass

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
# TODO: make fast for all data
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

# plots histogram of distances
# returns plot
def plot_distances(data_list):
	bins, edges = np.histogram(data_list, 50, normed=1)
	left,right = edges[:-1],edges[1:]
	X = np.array([left,right]).T.flatten()
	Y = np.array([bins,bins]).T.flatten()
	return plt.plot(X,Y)

# TODO: plot roc curve and find out how
def roc_plot_helper(genuine, imposter, thresh_dist):
	TPR = sum(genuine[np.where(genuine<thresh_dist)]) / float(sum(genuine))
	TNR = sum(imposter[np.where(imposter<thresh_dist)]) / float(sum(imposter))
	return (TPR, TNR)

def roc_plot(genuine, imposter):
	TPR = []
	TNR = []
	for thresh in range(4000):
		tpr, tnr = roc_plot_helper(genuine, imposter, thresh)
		TPR.append(tpr)
		TNR.append(tnr)
	return plt.plot(TNR, TPR)

# run me for question 1 part b
def partb():
	all_data = setup()
	display_each(all_data)

# run me for question 1 part c
def partc():
	all_data = setup()
	get_probs(all_data)

# run me for question 1 part d
def partd():
	all_data = setup()
	firstKNN(all_data)

def parte():
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

def partf():
	all_data = setup()
	genuine, imposter = binary_distances(all_data)
	plot = roc_plot(genuine, imposter)
	plt.set_xlabel("TNR")
	plt.set_ylabel("TPR")
	plt.show()

def partg():
	all_data = setup()
	kNN(all_data)
	
# partb()
# partc()
partd()
# parte()
# partf()
# partg()


