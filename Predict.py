# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:30:35 2018

@author: ishaa
"""
from pickle import load
from keras.models import load_model
from Model_handling import extract_features
from Model_handling import generate_desc

dataset_root_dir = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\'
code_root_dir = 'D:\\Study\\Machine Learning\\Codes\\Caption Generator\\Reverse-Image-Search\\'
weights = code_root_dir + 'Model Weights\\model_19.h5'
model = load_model(weights)

# load the tokenizer
tokenizer = load(open(code_root_dir + 'tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34


# load and prepare the photograph
photo = extract_features(dataset_root_dir + 'climbing.jpg')
# generate description
predicted_description = generate_desc(model, tokenizer, photo, max_length)
print(predicted_description)


testFile = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\Flickr_8k\\Flickr_8k.testImages.txt'
testImagesLabel = Load.load_set(testFile)

matchedFiles = set()

for img in testImagesLabel:
    actual, predicted = list(), list()
    yhat = predicted_description.split()
    predicted.append(yhat)
    references = [d.split() for d in test_descriptions[img]]
    actual.append(references) 
    bleu_score = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    if bleu_score > 0.5:
        matchedFiles.add(img)
        continue

len(matchedFiles)

path = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\Flicker8k_Dataset\\'

for img in matchedFiles:
    img_path = path + img + '.jpg'
    img2 = Image.open(img_path)
    img2.show()
    
matchedFiles    
