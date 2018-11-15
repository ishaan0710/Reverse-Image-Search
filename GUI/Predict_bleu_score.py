# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:54:01 2018

@author: ishaa
"""
import Loader
import string
from nltk.translate.bleu_score import corpus_bleu
from shutil import copyfile
import os


dataset_root_dir = 'D:\\Study\\Machine Learning\\DataSets\\Image Caption Generator\\'
code_root_dir = 'D:\\Study\\Machine Learning\\Codes\\Caption Generator\\Reverse-Image-Search\\'


input_file = open('C:\\xampp\\htdocs\\uploads\\description.txt', 'r')
predicted_description = input_file.readline()

table = str.maketrans('','',string.punctuation)

desc = predicted_description.split()
desc = [word.lower() for word in desc]
desc = [word.translate(table) for word in desc]
desc = [word for word in desc if len(word)>1]
desc = [word for word in desc if word.isalpha()]
predicted_description =  ' '.join(desc)

testFile = dataset_root_dir + 'Flickr_8k\\Flickr_8k.testImages.txt'
testImagesLabel = Loader.load_set(testFile)
test_descriptions = Loader.load_clean_descriptions(dataset_root_dir + 'descriptions.txt', testImagesLabel)

matchedFiles = set()

for img in testImagesLabel:
    actual, predicted = list(), list()
    yhat = predicted_description.split()
    predicted.append(yhat)
    references = [d.split() for d in test_descriptions[img]]
    actual.append(references) 
    bleu_score = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    if bleu_score > 0.5:
        matchedFiles.add(img)
        continue

len(matchedFiles)

path = 'D:/Study/Machine Learning/DataSets/Image Caption Generator/Flicker8k_Dataset/'

matched_img_file = open('C:\\xampp\\htdocs\\uploads\\matched_images.txt',"w")

folder = 'C:\\xampp\\htdocs\\uploads\\matched-images'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

for img in matchedFiles:
    img_path = path + img + '.jpg'
    matched_img_file.write(img_path + '\n')
    copyfile(img_path, folder + '\\' + img + '.jpg')
    
matchedFiles    
matched_img_file.close()
