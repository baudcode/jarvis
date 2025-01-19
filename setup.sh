# apt requirements
sudo apt-get install -y portaudio19-dev gcc libspeexdsp-dev


echo "Conda setup"
conda create -n wakeword python=3.11
conda activate wakeword

# install stdlib
conda install -c conda-forge libstdcxx-ng

# wakeword, wisperx requirements
pip install openwakeword pyaudio whisperx torch Cython sounddevice transformers librosa phonemizer unidecode
echo "Conda setup done"


echo "checkout vits and build extension"
git clone https://github.com/jaywalnut310/vits.git
mkdir -p vits/monotonic_align/monotonic_align/
cd vits/monotonic_align
python setup.py build_ext --inplace
cd ../../