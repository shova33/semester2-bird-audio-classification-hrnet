#https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_data.py

#from glob import glob
from random import choice
from time import time
import os
#import h5py
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split

path_to_data_train = "E:/RUNNING_PROJECTS/WILD_LIFE_DB_SABDA/Wildlife_Datasets/Processed_Data/Train/" #Label as string
h5_data_train = []
path_to_data_test = "E:/RUNNING_PROJECTS/WILD_LIFE_DB_SABDA/Wildlife_Datasets/Processed_Data/Val/" #Label as string
h5_data_test = []

#Training Data
for key in os.listdir(os.path.join(path_to_data_train, "Bear")):
    h5_data_train.append(os.path.join(path_to_data_train, "Bear", key))

for key in os.listdir(os.path.join(path_to_data_train, "Cat")):
    h5_data_train.append(os.path.join(path_to_data_train, "Cat", key))
for key in os.listdir(os.path.join(path_to_data_train, "African", "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "African", "Positive", "audio", key))

for key in os.listdir(os.path.join(path_to_data_train, "Arabic",  "Negative","audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Arabic",  "Negative", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Arabic",  "Neutral","audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Arabic",  "Neutral", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Arabic",  "Positive","audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Arabic",  "Positive", "audio", key))

for key in os.listdir(os.path.join(path_to_data_train, "Chinese",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Chinese",  "Negative", "audio", key))
    
     
for key in os.listdir(os.path.join(path_to_data_train, "English",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "English",  "Negative", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "English",  "Neutral", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "English",  "Neutral", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "English",  "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "English",  "Positive", "audio", key))    
    
for key in os.listdir(os.path.join(path_to_data_train, "French",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "French",  "Negative", "audio", key)) 
for key in os.listdir(os.path.join(path_to_data_train, "French",  "Neutral", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "French",  "Neutral", "audio", key)) 
for key in os.listdir(os.path.join(path_to_data_train, "French",  "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "French",  "Positive", "audio", key)) 

for key in os.listdir(os.path.join(path_to_data_train, "Indian",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Indian",  "Negative", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Indian",  "Neutral", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Indian",  "Neutral", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Indian",  "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Indian",  "Positive", "audio", key))
    
for key in os.listdir(os.path.join(path_to_data_train, "Nepali",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Nepali",  "Negative", "audio", key)) 
for key in os.listdir(os.path.join(path_to_data_train, "Nepali",  "Neutral", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Nepali",  "Neutral", "audio", key)) 
for key in os.listdir(os.path.join(path_to_data_train, "Nepali",  "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Nepali",  "Positive", "audio", key)) 

for key in os.listdir(os.path.join(path_to_data_train, "Spanish",  "Negative", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Spanish",  "Negative", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Spanish",  "Neutral", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Spanish",  "Neutral", "audio", key))
for key in os.listdir(os.path.join(path_to_data_train, "Spanish",  "Positive", "audio")):
    h5_data_train.append(os.path.join(path_to_data_train, "Spanish",  "Positive", "audio", key))
    
shuffle(h5_data_train)
shuffle(h5_data_test)

#https://www.kaggle.com/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras
def normalize(img):
    '''
    Normalizes an array 
    (subtract mean and divide by standard deviation)
    '''
    eps = 0.001
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img


class DataReader:
    def __init__(self):
        #self.train_files = h5_data #glob(os.path.join(OUTPUT_DIR_TRAIN, '**.h5'))
        self.train_files, self.val_files = train_test_split(sorted(h5_data_train), test_size=0.05, random_state=42)
        self.train_files, self.test_files = train_test_split(sorted(self.train_files), test_size=0.01, random_state=42)
        
        print('Training files =', len(self.train_files))
        print('Validation files =', len(self.val_files ))
        print('Testing files =', len(self.test_files))

    def next_batch_train(self, batch_size):
        return DataReader._next_batch(batch_size, self.train_files)
    
    def next_batch_test(self, batch_size):
        return DataReader._next_test_batch(batch_size, self.test_files)
    
    def test_files_count(self):
        return len(self.test_files)
    
    def train_files_count(self):
        return len(self.train_files)
    
    def val_files_count(self):
        return len(self.val_files)

    def get_all_training_data(self):
        return DataReader._get_data(self.train_files)
    
    def get_all_validation_data(self):
        return DataReader._get_data(self.val_files)
    
    def get_all_test_data(self):
        return DataReader._get_test_data(self.test_files)
    
    
    
    def generator_train(self, batch_size):
        while True:
            melgram, target  = DataReader._next_batch(batch_size, self.train_files)
            yield melgram, target
    
    def generator_val(self, batch_size):
        while True:
            melgram, target  = DataReader._next_batch(batch_size, self.val_files)
            yield melgram, target
    
    def generator_test(self, batch_size):
        while True:
            melgram, target  = DataReader._next_test_batch(batch_size, self.test_files)
            yield melgram, target
    
    
    @staticmethod
    def _get_test_data(file_list):
        
        def load_into(filename, melgram, target):
            #print(filename)
            loadeddata = np.load(filename)   
            melgram.append(normalize(loadeddata["melgram"].astype(np.int32)))
            target.append(loadeddata["target"].astype(np.int32))
            
        melgram, target =  [], []
        for filename in file_list:
            load_into(filename, melgram, target)
        
        return np.array(melgram), np.array(target)
        
    @staticmethod
    def _get_data(file_list):
        
        def load_into(filename, melgram, target):
            #print(filename)
            loadeddata = np.load(filename)   
            melgram.append(normalize(loadeddata["melgram"].astype(np.int32)))
            target.append(loadeddata["target"].astype(np.int32))
            
        melgram, target =  [], []
        for filename in file_list:
            load_into(filename, melgram, target)
        
        return np.array(melgram), np.array(target) 
    
    @staticmethod
    def _next_test_batch(batch_size, file_list):
        return DataReader._get_test_data([choice(file_list) for _ in range(batch_size)])
    
    @staticmethod
    def _next_batch(batch_size, file_list):
        return DataReader._get_data([choice(file_list) for _ in range(batch_size)])


if __name__ == '__main__':
    data_reader = DataReader()
    a = time()
    for i in range(5):
        #print(i)
        melgram, target= data_reader.next_batch_train(10)
        print('melgram shape =', melgram.shape)
        
    print(time() - a, 'sec')
    #print('The tain batch:', train)
    
    print('target =', target, target.shape)
    print('mel_phasegram.shape =', melgram.shape)

    
