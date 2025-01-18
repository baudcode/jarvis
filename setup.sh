# apt requirements
sudo apt-get install -y portaudio19-dev gcc libspeexdsp-dev

# conda environment setup
conda create -n wakeword python=3.11
conda activate wakeword

# install stdlib
conda install -c conda-forge libstdcxx-ng

# wakeword, wisperx requirements
pip install openwakeword pyaudio whisperx torch Cython sounddevice transformers librosa phonemizer unidecode
# pip install openai-whisper
