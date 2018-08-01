#python3 -m pip install pydub
#python3 -m pip install librosa
#yum install ffmpeg libavcodec-extra-5

import IPython.display as ipd
import librosa

from pydub import AudioSegment
sound = AudioSegment.from_mp3("/data/urban_sound/suk_eki_labonnye_RabindroSangeet.mp3")
sound.export("/data/urban_sound/suk_eki_labonnye_RabindroSangeet.wav", format="wav")
data_set, sampling_rate_sett = librosa.load('/data/urban_sound/suk_eki_labonnye_RabindroSangeet.wav')

print(data_set)

print(sampling_rate_sett)


% pylab inline
import os
import pandas as pd
import librosa
import glob
import librosa.display

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data_set, sr=sampling_rate_sett)


features = np.mean(librosa.feature.mfcc(y=data_set, sr=sampling_rate_sett, n_mfcc=40).T,axis=0)
label = 'suk_sett'
