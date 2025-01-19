import sys
from os.path import dirname, join

sys.path.append(join(dirname(dirname(__file__)), "vits")) # noqa

import IPython.display as ipd

from timeit import default_timer
import torch

import commons
import utils

from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from pathlib import Path
import tempfile

DEFAULT_CHECKPOINT_PATH = "vits/pretrained_ljs.pth"
DEFAULT_CONFIG_PATH = "vits/configs/ljs_base.json"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def sampels2wav(samples, rate=16_000):
    audio_file = ipd.Audio(samples, rate=rate, normalize=False)
    return audio_file.data

class AudioGeneratorModel:

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        self.hps = utils.get_hparams_from_file(config_path)

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)
    
    @property
    def rate(self):
        return self.hps.data.sampling_rate
    
    def __call__(self, text: str):

        start_time = default_timer()
        print(f"=> generating audio for {text=}")

        for char in ['{', '}', '(', ')']:
            text = text.replace(char, ",")

        stn_tst = get_text(text, self.hps)

        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

        elapsed = default_timer() - start_time
        print(f"=> audio generated in {elapsed:.2f}s")
        return audio
    

    def generate_wav(self, text: str, output_file: Path):
        audio = sampels2wav(self(text), self.rate)

        with Path(output_file).open("wb") as writer:
            writer.write(audio)
        
        print(f"=> generated {output_file=}")

    
    def display(self, text: str):
        filename = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        self.generate_wav(text, Path(filename))
        return ipd.Audio(filename=filename)


def play(data, rate=16_000):
    import sounddevice as sd
    import soundfile as sf

    # Replace 'audio.wav' with the path to your audio file
    # data, fs = sf.read('audio.wav', dtype='float32')
    sd.play(data, rate)
    sd.wait()

def play_file(filename):
    import sounddevice as sd
    import soundfile as sf

    # Replace 'audio.wav' with the path to your audio file
    data, fs = sf.read(filename, dtype='float32')
    sd.play(data, fs)
    sd.wait()



def test():

    model = AudioGeneratorModel()
    # model.display("You are awesome!")
    model.generate_wav(
        "You are awesome! (Chad!)", "awesome.wav"
    )
    play_file("awesome.wav")



if __name__ == "__main__":
    test()
