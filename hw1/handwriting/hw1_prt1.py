import csv 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import scipy.spatial.distance as sp
import scipy

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
def mainKNN(all_data):
	data_no_digits, label, digits = find_each(all_data)
	digits = scipy.array(digits).astype(int)
	no_dig_label,no_dig_data = modify_structure(data_no_digits)
	for k in range(len(digits)):
		distance = []
		for i in range(len(no_dig_data)):
			distance.append(sp.euclidean(no_dig_data[i],digits[k]))
		mindex=np.argmin(np.array(distance))
		if(int(no_dig_label[mindex]) == int(label[k])):
			print "actual",label[k], ", label of nn", no_dig_label[mindex] 
		else:
			print "actual",label[k], ", label of nn", no_dig_label[mindex], "*"

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

# run me for question 1 part b
def partb():
	all_data=setup()
	display_each(all_data)

# run me for question 1 part c
def partc():
	all_data=setup()
	get_probs(all_data)

# run me for question 1 part d
def partd():
	all_data=setup()
	mainKNN(all_data)

# partb()
# partc()
partd()


