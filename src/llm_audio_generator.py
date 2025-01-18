from .audio_generator import AudioGeneratorModel, play
from .realtime_audio import RealtimeTranscription
from .llm import sentence_iterator, query_model

from threading import Thread
import time

class LLMAudioGenerator:
    def __init__(self, audio_gen: AudioGeneratorModel):
        self.audio_gen = audio_gen

    def query(self, question: str):
        
        gen = query_model(question)

        play_thread: Thread = None

        for sentence in sentence_iterator(gen):
            print(f"=> llm respponse sentence: {sentence}")
            samples = self.audio_gen(sentence)
            print("=> playing sample")
            # start thread to play

            # dont start playing if other play is still in progress
            while play_thread and play_thread.is_alive():
                time.sleep(.01)

            play_thread = Thread(target=play, args=(samples, self.audio_gen.rate))
            play_thread.start()
            
def main():
    
    llm = LLMAudioGenerator(AudioGeneratorModel())
    rt = RealtimeTranscription()
    rt.register_callback(llm.query)
    
    # run on audio
    rt()


if __name__ == "__main__":
    main()