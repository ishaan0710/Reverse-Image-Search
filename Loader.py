# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:43:09 2018

@author: ishaa
"""

from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

'''
def get_session(gpu_fraction=0.8):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())

'''
#Loading image identifiers


#Loading cleaned descriptions
#Load photo features

#Loading text from file
def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()    
    return text

#Image Identifier --- TrainImage.txt
    #2513260012_03d33305cf.jpg
    #2903617548_d3e38d7f88.jpg
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
    
#load cleaned description
#1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way
#assign start and end token to the description
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0],tokens[1:]
        #if image_id not in dataset ignore
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions
    
# load photo features from features.pkl
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

#convert descriptions into a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(des) for des in descriptions[key]]
    return all_desc

#Encode using tokenizer
#eg - tokenize  --> Boy on horse --> startseq Boy on horse endSeq
# X1      X2                        y (word)
# photo   startseq                  Boy
# photo   startseq, Boy             on
# photo   startseq, Boy,on          horse
# photo   startseq,Boy,on,horse     endseq
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer    

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

#split into X1 X2 and y
""""def create_sequences(tokenizer, max_length, descriptions, photos):   
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            #convert tokenized text into sequence
            
            seq = tokenizer.texts_to_sequences([desc])[0]
    			# split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
    				# split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
    				# pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
    				# encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
    				# store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)
"""

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
    
def define_model(vocab_size, max_length):
    #feature extractor model
    inputs1  = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation = 'relu')(fe1)
    #Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #decoder model
    decoder1 = add([fe2,se3])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]
            
#mapping of integer to word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    #start seq
    in_text = 'startseq'
    #repeatedly generate next letter using created in_text until it is not 'endseq' 
    for i in range(max_length):
        #convert to tokens
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #padding
        sequence = pad_sequences([sequence], maxlen = max_length)
        #find probability
        yhat = model.predict([photo, sequence], verbose=0)
        #how?
        yhat = argmax(yhat)
        #map integer to word
        word = word_for_id(yhat, tokenizer)
        print(sequence)
        print(yhat)
        print(word)
        if word is None:
            break
        in_text += ' ' + word 
        if word == "endseq":
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    
    for key, desc_list in descriptions.items():
            #generate description
            yhat = generate_desc(model, tokenizer, photos[key], max_length)
            # store actual and predicted captions
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
                        
# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('D:\\Study\\Machine Learning\\Codes\\Caption Generator\\Reverse-Image-Search\\model_' + str(i) + '.h5')
            
    
    
#load trainset
trainFile = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\Flickr_8k\\Flickr_8k.trainImages.txt'
train = load_set(trainFile)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
"""
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset
 
# load test set
filename = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\Flickr_8k\\Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
 
# fit model
 
# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest)
"""

# load test set
filename = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\Flickr_8k\\Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\features.pkl', test)
print('Photos: test=%d' % len(test_features))

for i in range(0,20):
    filename = 'D:\\Study\\Machine Learning\\Codes\\Caption Generator\\Reverse-Image-Search\\Model Weights\\model_' + str(i) + '.h5'
    model = load_model(filename)
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
    print('\n---------------------------------------------------------------------------------------------\n')

filename = 'D:\\Study\\Machine Learning\\Codes\\Caption Generator\\Reverse-Image-Search\\Model Weights\\model_19.h5'
model = load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)





def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

from pickle import dump

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model(filename)
# load and prepare the photograph
photo = extract_features('D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\cat.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)





