from __future__ import print_function
from __future__ import absolute_import

#import time
import tqdm
import os
import numpy as np
from glob import glob
import h5py as h5

import librosa

import warnings
warnings.filterwarnings('ignore')


#********************AUDIO PROCESSING setting********************#  
audio_duration = 5   
sample_rate = 32000 #16000
window_size = 1024 # or frame size
hop_size = 320     # So that there are 64 frames per second
mel_bins = 64

frames_per_sec = sample_rate // hop_size 
frames_num = frames_per_sec * audio_duration #Total temporal frames = 64*10 =640    
audio_samples = int(sample_rate * audio_duration)


genre_calss = ["Negative", "Neutral", "Positive"] # Class name here


#*******************General functions *****************************************# 

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

#Class encoding (One-hot)
def encode_class(class_name, class_names):  
    try:
        idx = class_names.index(class_name) #If there is string input then: class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None

def normalize(audio):
    """
    Normalizes an array 
    (subtract mean and divide by standard deviation)
    """
    eps = 0.001
    if np.std(audio) != 0:
        audio = (audio - np.mean(audio)) / np.std(audio)
    else:
        audio = (audio - np.mean(audio)) / eps
    return audio


def normalize_centre_and_amplitude(audio):
    """
    # center and normalize amplitude
    """
    #audio = audio.astype(np.float32)
    samples = audio - audio.mean()
    max_amplitude = np.max([-samples.min(), samples.max()])
    samples = samples / max_amplitude
    
    return samples
    

#*******************AUDIO PROCESSING: Basic functions *****************************************# 

def read_audio(audio_path, target_fs=None):
    #(audio, fs) = soundfile.read(audio_path)
    audio, fs = librosa.load(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs

#Repeat audio waveform until defined length
def repeat_array(array, max_len):  
    random_state = np.random.RandomState(1234)
    
    repeats_num = max_len // len(array) + 1
    
    repeated_array = []
    
    for n in range(repeats_num):
        random_state.shuffle(array)
        repeated_array.append(array)
        
    repeated_array = np.concatenate(repeated_array, axis=0)
    repeated_array = repeated_array[0 : max_len]

    return repeated_array


def normalize_to_energy(x, db):
    return x / np.sqrt(np.mean(np.square(x))) * np.power(10., db / 20.)

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

#*******************AUDIO PROCESSING: Log Mel-spectrogram *****************************************#  

def compute_mel_spec(audio_name):    
                
    # Compute short-time Fourier transform
    stft_matrix = librosa.core.stft(y=audio_name, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins).T
    # Mel spectrogram
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)
    # Log mel spectrogram
    logmel_spc = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spc = np.expand_dims(logmel_spc, axis=0)
    logmel_spc = logmel_spc.astype(np.float32)
    #print("The shape of logmel_spc:", logmel_spc.shape)
    logmel_spc = np.array(logmel_spc).transpose((2, 1, 0))
    logmel_spc = logmel_spc[0 : frames_num]
    
    
    return logmel_spc

#***********************************************************Audio PROCESSING START *****************************************#  
def audio_feature(audio_path):
    # Read audio
    (audio, _) = read_audio(audio_path=audio_path, target_fs=sample_rate)
    #multichannel_audio, fs = read_multichannel_audio(audio_path=audio_path, target_fs=sample_rate)
    
    if audio.shape[0] < audio_samples: 
        #audio = repeat_array(audio, audio_samples) #Repeat the same audio until 10sec length
        audio = pad_truncate_sequence(audio, audio_samples) ## Pad or truncate audio recording
    
    #If audio length is more than 10second then clip it to 10Sec
    elif audio.shape[0] > audio_samples:
        audio = audio[int((audio.shape[0]-audio_samples)/2):int((audio.shape[0]+audio_samples)/2)]
    
    logmel_spc = compute_mel_spec(audio)
    #print("The mel_phasegram shape is:",mel_phasegram.shape)
    
    return normalize(logmel_spc)
    # or return normalize_centre_and_amplitude(logmel_spc), normalize_centre_and_amplitude(phase_gram)


def main(root_directory, output_directory, file_path):

    file_audio = output_directory + os.path.sep + file_path[len(root_directory):] 
    base_file = os.path.splitext(file_audio)[0]  #Remove the .mp4 extension
    #hdf5_file = base_file + '.h5'
    npz_file = base_file + '.npz'
    
    if os.path.exists(npz_file):
        return True

    os.makedirs(os.path.dirname(npz_file), exist_ok=True)
    
    #********************** For target generation ***********************************#
    # encode the class labels using one hot coding (use index as label)
    
    #Making Folder as Label
    basepath, _ = os.path.split(file_path)
    sub_dir, _ = os.path.split(basepath)
    _, label = os.path.split(sub_dir)
    print("The current filename is:", label)

    target = encode_class(label, genre_calss)
    #print("The label_onehot shape is:",label_onehot.shape)    
    
    #**************************Mix Audio processing***************************************************#
    logmel_spc  = audio_feature(file_path)
    print("The logmel_spc shape is:", logmel_spc.shape)

    np.savez(npz_file, melgram=logmel_spc, target = target)
    
    
if __name__ == '__main__':
    
    audio_directory = 'E:/RUNNING_PROJECTS/WILD_LIFE_DB_SABDA/Wildlife_Datasets/Audio Dataset/Train/'
    output_directory = 'E:/RUNNING_PROJECTS/WILD_LIFE_DB_SABDA/Wildlife_Datasets/Processed_Data/Train/'
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    
    for root_dir in tqdm.tqdm(sorted(list(glob(os.path.join(audio_directory, '*')))), position=0):
        for sub_dir in tqdm.tqdm(sorted(list(glob(os.path.join(root_dir, '*')))), position=1, leave=False):
            for file_path in tqdm.tqdm(list(glob(os.path.join(sub_dir, '*.wav'))), position=2, leave=False):
                #print("The input utterance dir", utterance)
                print("The input utterance dir", file_path)
                main(audio_directory, output_directory, file_path)       
