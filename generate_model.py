import numpy as np
import tensorflow as tf
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

EMBEDDING_DIM = 256 #size for embedding(embedding is like fixing firmly)

lstm_layers = 2		#for guide
dropout_rate = 0.5 #for guide
learning_rate = 0.001 #the value which is multiplied with gradient to minimize loss

#convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc=list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

#fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines=to_lines(descriptions)
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines) #keras function
	return tokenizer



#calculate the length of the description with the most words
def max_length(descriptions):
	lines=to_lines(descriptions)
	return max(len(d.split()) for d in lines)

#create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer,max_length,desc_list,photo):
	vocab_size=len(tokenizer.word_index)+1 #it gives the vacabulary size and tokenizer.word_idex is keras function

	X1, X2, y = [], [], [] #x1 for image id, x2 for image description 
	#walk through each description for the image
	for desc in desc_list:
		#encode the sequences
		seq = tokenizer.texts_to_sequences([desc])[0] #vayeko description lai sequence ma lageko i.e. encoding seq
		#split one sequence into multiple X,y pairs
		for i in range(1,len(seq)): #split seq into multi X,y pairs
			#split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]  #desc input-output pair
			#pad input sequence (pad means to make the size same)
			in_seq=pad_sequences([in_seq], maxlen=max_length)[0] #making the size same for all dimensions
			#encode output sequence
			out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
			#store
			X1.append(photo) #appending  img IDs
			X2.append(in_seq) #multi X-y pairs encoding
			y.append(out_seq) # encoded version of output word
	return np.array(X1), np.array(X2),np.array(y)

#data generator, intended to be used in ca call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer,max_length, n_step=1):
	#loop for ever over images
	while 1:
		#loop over photo identifiers in the dataset
		keys = list(descriptions.keys())
		for i in range(0,len(keys),n_step):
			Ximages, XSeq, y = list(), list(), list()
			for j in range(i, min(len(keys), i+n_step)):
				image_id=keys[j]
				#retrieve the photo feature
				photo=photos[image_id][0]
				desc_list=descriptions[image_id]
				in_img, in_seq, out_word=create_sequences(tokenizer,max_length,desc_list,photo)
				for k in range(len(in_img)):
					Ximages.append(in_img[k])
					XSeq.append(in_seq[k])
					y.append(out_word[k])
			yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]



#define the captioning model
def define_model(vocab_size, max_length):
	#feature extractor (encoder)
	#Encoder Models (Img-Feat and Desc Encoding)
	firstinput = Input(shape=(4096,))
	feat1=Dropout(0.5)(firstinput)
	feat2=Dense(EMBEDDING_DIM, activation='relu')(feat1)
	feat3=RepeatVector(max_length)(feat2)

	#Embedding+LSTM sequence model
	description_input = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(description_input) #vocab_size is the number of words that we have,emedding dim is the size for embedding
	#merge inputs
	merged=concatenate([feat3,emb2])
	#language model(decoder)
	lm2=LSTM(500, return_sequences=False)(merged) #500 means 500 outputs
	outputs=Dense(vocab_size,activation='softmax')(lm2) 

	#tie it toegether [image, seq] [word],Creating Model-Input-Output architecture;
	model=Model(inputs=[firstinput, description_input], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model