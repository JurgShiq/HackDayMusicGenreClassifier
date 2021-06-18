import librosa
import subprocess
import matplotlib.pyplot as plt
import librosa.display

audio_path = '/Users/jurgshiq/Documents/GitHub/HackDayMusicGenreClassifier/music/SGLewisAura.mp3'

x , sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()
#return_code = subprocess.call(["afplay", audio_path])