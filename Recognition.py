import pprint
from random import sample
import re
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import sys
import pickle
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from IPython.display import Audio
from tensorflow import keras

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    print("ZCR ========> ",zcr)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    print("Chroma STFT ========> ",chroma_stft)
    # print(len(chroma_stft)) #12
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    print("MFCC ========> ",mfcc)
    # print(len(mfcc)) #20
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    print("RMS ========> ",rms)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    # print("MEL ========> ",mel)
    # print(len(mel)) #128
    result = np.hstack((result, mel)) # stacking horizontally

    print("==================\n\n")

    print(result[0])   #Zero Crossing Rate (multiply 100)
    print(result[1:13])   #Chroma STFT -- 12 values
    print(result[13:33])  #MFCC -- 20 values
    print(result[33])    #Root Mean Square (multiply 100)
    
    print("==================")
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    # print("===============NOISE DATA FEATURES==============")
    res2 = extract_features(noise_data)
    # print(res2)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


Y = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

src = '/Users/divya/Desktop/happy.wav'

data, sample_rate = librosa.load(src)
print(sample_rate,"RATEEE") 
Feature_list = get_features(src)

scaler  = pickle.load(open('scaler.pkl','rb'))
loaded_model = load_model("savedmodel.h5")

# print("Not scaled Features --> ",Feature_list)
scaled_features = scaler.transform(Feature_list)
# print("Scaled Features --> ",scaled_features)

scaled_features = pd.DataFrame(scaled_features)
Features = np.expand_dims(scaled_features, axis=2)
# print(Features.shape)

predicted_feature = loaded_model.predict(Features)
# print(predicted_feature)

y_pred = encoder.inverse_transform(predicted_feature)

print("\n\nPrediction using features infiltrated with noise -> ",y_pred.flatten()[0])
print("Prediction using stretched features -> ",y_pred.flatten()[1])
print("Prediction using pitched features -> ",y_pred.flatten()[2])
print("\n\n")

Feature_list = []