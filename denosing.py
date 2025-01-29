import soundfile as sf
import noisereduce as nr
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt

# Load the audio file
audio_data, sample_rate = sf.read('amstrong.ogg')

# Create a figure and plot the audio waveform before processing
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data, color='blue')
plt.title("Audio Waveform (Before Processing)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig('audio_waveform_before.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Extract MFCC features from the audio before processing
mfcc_features_before = mfcc(audio_data, sample_rate)

# Apply noise reduction using the noisereduce library
enhanced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

# Save the enhanced audio as a WAV file
sf.write('enhanced_audio.wav', enhanced_audio, sample_rate)

# Extract MFCC features from the enhanced audio
mfcc_features_after = mfcc(enhanced_audio, sample_rate)

# Create a figure and plot the audio waveform after processing
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(enhanced_audio) / sample_rate, len(enhanced_audio)), enhanced_audio, color='green')
plt.title("Audio Waveform (After Processing)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig('audio_waveform_after.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Save MFCC features as images
plt.figure(figsize=(10, 4))
plt.imshow(mfcc_features_before, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar()
plt.title("MFCC Features (Before Processing)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency (Hz)")
plt.savefig('mfcc_features_before.png', bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(figsize=(10, 4))
plt.imshow(mfcc_features_after, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar()
plt.title("MFCC Features (After Processing)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency (Hz)")
plt.savefig('mfcc_features_after.png', bbox_inches='tight', pad_inches=0)
plt.close()

print("Enhanced audio saved as enhanced_audio.wav")
print("MFCC features (before processing) saved as mfcc_features_before.png")
print("MFCC features (after processing) saved as mfcc_features_after.png")
print("Audio waveform (before processing) saved as audio_waveform_before.png")
print("Audio waveform (after processing) saved as audio_waveform_after.png")