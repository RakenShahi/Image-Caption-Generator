from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
from progressbar import progressbar
from keras.models import Model

#path bata image load hanni
def image_load(path): #yesma chai chaini image lai shape ma lyauni work
	imge=load_img(path,target_size=(224,224)) #image loading having 224,224 as dimension
	imge=img_to_array(imge) #image lai array ma lageko
	imge=np.expand_dims(imge,axis=0) #axis is a position where new axis is to be inserted
	imge=preprocess_input(imge) #function availabe in vgg16 for preprocessing of image
	return np.asarray(imge) #np.asarray would reflect the changes in the original array.

#extract features from each photo in the directory
def feature_extract(directory,attn=False):
	
	#load the model
	if attn:
		modelfile=VGG16()
		modelfile.layers.pop() #Restructing the model (removing the last softmax layer; retain the penultimate FCC-4096)
		#extract final 49*512 conv layer for context vectors
		fnl_con=Reshape([49,512])(modelfile.layers[-4].output)  #taking transfer layer
		modelfile=Model(inputs=modelfile.inputs, outputs=fnl_con)
		print(modelfile.summary())
		features=dict() #it is the data collection i.e. features collection it contains keys and values,key=input,value=output
	else:
		modelfile=VGG16()
		#re-structure the model
		modelfile.layers.pop()
		modelfile=Model(inputs=modelfile.inputs,outputs=modelfile.layers[-1].output)
		print(modelfile.summary())
		features=dict()

	for name in progressbar(listdir(directory)):
		FN = directory + '/' + name #patho for our image directory
		image=image_load(FN) #load our imge ani image ma chaini shape ma returned value aaux
		#extract features of our image
		feature=modelfile.predict(image,verbose=0) #verbose is used to show the training progress, verbose=0 means showing nothing
		#get image id
		image_id=name.split('.')[0] #filename lai . le chutayera first part lini.[0] means first
		#store feature
		features[image_id]=feature #Dictionary ma image ko feature store gareko with their image_id #mapping and storing image_features to image_id 
		print('>%s' % name)
	return features


#load doc into memory
def load_doc(filename):
	#open the file as read only
	file=open(filename,'r')
	#read all text
	text=file.read()
	#close the file
	file.close()
	return text

#extract descriptions for images
def load_descriptions(doc):
	mapping=dict()
	#process lines
	for line in doc.split('\n'): #lines chutauni and loops on every single line
		#split the lines by white space
		tokens=line.split() #spilting the token by middle white space of image id and caption
		if len(line)<2: #skipping blank lines
			continue
		#take the first token as the image id, the rest as the description
		image_id, image_desc=tokens[0], tokens[1:] #first portion as image id ani [1:] means baki sabai desciption
		#remove filename (extension) from image id
		image_id=image_id.split('.')[0]
		#convert description tokens back to string
		image_desc = ' '.join(image_desc)
		#create the list if needed
		if image_id not in mapping:
			mapping[image_id]=list()
		#store description
		mapping[image_id].append(image_desc) #here at first id is saved to dictionary and list is made to store the desciption
	return mapping

def cleaned_description(desciption):
	#prepare translation table for removing punctuation
	table=str.maketrans('','',string.punctuation) #it removes all the punctuation
	for key, desc_list in desciption.items(): #it gives both key and value in each step of iteration
		for i in range(len(desc_list)):
			desc=desc_list[i]
			#tokenize
			desc=desc.split() #each word are splitted and saved as arrays eg: d="ab" gives ['a','b']
			#convert to lower case
			desc=[word.lower() for word in desc]
			#remove punctuation from each token
			desc=[w.translate(table) for w in desc] # w as a string mathi table value ma jancha ani punctuations haru remove hunx
			#remove hanging 's' and 'a'
			desc=[word for word in desc if len(word)>1]
			#remove tokens with numbers in them
			desc=[word for word in desc if word.isalpha()]
			#store as string
			desc_list[i]= ' '.join(desc)

#convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
  # build a list of all description strings
  desc_all = set()
  for key in descriptions.keys():
    [desc_all.update(d.split()) for d in descriptions[key]]#fill the set with vocabularies from description
  return desc_all

#save descriptions to file, one per line
def save_descriptions(descriptions,filename):
	lines=list()
	for key,desc_list in descriptions.items(): #accessing both keys and values in each loop
		for desc in desc_list:
			lines.append(key + ' ' + desc )
	data ='\n'.join(lines)
	createfile=open(filename,'w')
	createfile.write(data)
	createfile.close()

#extract features from all images
directory='Flickr8k_Dataset'
features=feature_extract(directory)
print ('Extracted Features: %d' % len(features))
#save to file
dump(features,open('models/features.pkl','wb')) #we receive a feature cached file having 4096 elements


#prepare descriptions
filename='Flickr8k_text/Flickr8k.token.txt'
#load descriptions
doc=load_doc(filename)
#parse descriptions
descriptions=load_descriptions(doc)
print('Loaded: %d' % len(descriptions))
#clean descriptions
cleaned_description(descriptions)
#summerize vocabulary
vocab=to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocab))
#save to file
save_descriptions(descriptions,'models/descriptions.txt')

