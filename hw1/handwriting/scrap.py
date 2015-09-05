import csv, Image
import matplotlib.pyplot as plt 
import numpy as np

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

def display_each(all_data):
	seen_data=[]
	seen_nums=[]
	for el in all_data:
		if not(el[0] in seen_nums):
			seen_nums.append(el[0])
			seen_data.append(el[1])
		if len(seen_nums)==10:
			break

	for index in range(len(seen_nums)):
		print(seen_nums[index])
		make_image(seen_data[index])

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

data_file=open("train.csv","rb")
data=csv.reader(data_file,delimiter=",")
digits=[]
index=0
for row in data:
	if index!=0:
		digits.append((row[0],row[1:]))
	index+=1
get_probs(digits)
# make_image(digits[7][1])
# display_each(digits)


# print digits[0]
# make_image(digits[0])


