from openwakeword import Model as OpenWakeWordModel
import whisperx

import sounddevice as sd
import numpy as np
from collections import deque
from timeit import default_timer
from typing import Callable, Literal

class RealtimeTranscription:

    def __init__(self, transcription_model="small.en", device='cpu', rate=16_000, buffer_duration=5, wakeword_model="hey_jarvis"):

        # Initialize Whisper model
        # model = whisper.load_model("base")
        self.transcription_model_name = transcription_model
        self.transcription_model = whisperx.load_model(transcription_model, device, compute_type="float32")
        self.buffer_duration = buffer_duration
        self.rate = rate
        self.buffer_size = self.buffer_duration * self.rate

        # Create a ring buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.state: Literal['wake_detection', 'tracking', 'idle'] = 'wake_detection'
        self.block_size = 1280

        self.wakeword_model_name = wakeword_model
        self.wakeword_model = OpenWakeWordModel(inference_framework="onnx", wakeword_models=[wakeword_model])
        self.callbacks: list[Callable[[str], None]] = []


    def audio_callback(self, indata, frames, time, status):

        if status:
            print("Status:", status)

        buf = indata[:, 0] # int16

        if self.state == 'tracking':
            # Append new audio to the buffer
            audio_samples = buf.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            self.audio_buffer.extend(audio_samples)
        elif self.state == 'wake_detection':
            # Feed to openWakeWord model
            prediction = self.wakeword_model.predict(buf)['hey_jarvis']

            print("oww prediction: ", prediction)
            if prediction >= 0.5:
                print("keyword detected")
                self.state = 'tracking'
    
    def register_callback(self, callback: Callable[[str], None]):
        self.callbacks.append(callback)
    

    def process_audio(self):
        """Processes audio from the buffer in overlapping chunks."""

        print("Listening...")

        while True:

            # Wait until there's enough data in the buffer
            if len(self.audio_buffer) < self.audio_buffer.maxlen:
                continue

            self.state = 'idle'

            print(f"=> Run predictions on {self.transcription_model_name}")
            start_time = default_timer()
            result = self.transcription_model.transcribe(np.array(list(self.audio_buffer), "float32"), batch_size=32)
            segments = result.get('segments', [])

            if len(segments) != 0:
                text = " ".join(map(lambda x: x['text'], segments))
                elapsed = default_timer() - start_time
                print(f"=> Transription ({self.transcription_model_name}): {text} [in {elapsed}s]")

                for callback in self.callbacks:
                    callback(text)
            
            self.state = 'wake_detection'

    def __call__(self):
        try:
            # Start the audio stream
            with sd.InputStream(samplerate=self.rate, blocksize=self.block_size, dtype=np.int16, channels=1, callback=self.audio_callback):
                self.process_audio()
        except KeyboardInterrupt:
            print("\nStopped listening.")

if __name__ == "__main__":
    audio = RealtimeTranscription()
    audio()
