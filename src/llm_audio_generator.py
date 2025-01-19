from .audio_generator import AudioGeneratorModel, play
from .realtime_audio import RealtimeTranscription
from .llm import LLM, OLLAMA_API_URL

from . import llm, audio_generator, realtime_audio

from threading import Thread
import time
import dataclasses



class LLMAudioGenerator:
    def __init__(self, audio_gen: AudioGeneratorModel, llm: LLM):
        self.audio_gen = audio_gen
        self.llm = llm

    def query(self, question: str):

        question = "Do not repeat the question! Answer precisely in two or three sentences the following question: " + question       
        gen = self.llm.query_sentences(question)

        play_thread: Thread = None

        for sentence in gen:
            print(f"=> llm respponse sentence: {sentence}")
            samples = self.audio_gen(sentence)
            print("=> playing sample")
            # start thread to play

            # dont start playing if other play is still in progress
            while play_thread and play_thread.is_alive():
                time.sleep(.01)

            play_thread = Thread(target=play, args=(samples, self.audio_gen.rate))
            play_thread.start()
        
        if play_thread:
            play_thread.join()


@dataclasses.dataclass
class Args:
    audio_generator_checkpoint_path: str = audio_generator.DEFAULT_CHECKPOINT_PATH
    audio_generator_config_path: str = audio_generator.DEFAULT_CONFIG_PATH
    ollama_api_url: str = llm.OLLAMA_API_URL
    llm: str = llm.DEFAULT_MODEL
    transcription_model: str = realtime_audio.DEFAULT_TRANSCRIPTION_MODEL
    buffer_duration: int = realtime_audio.DEFAULT_BUFFER_DURATION
    device: str = "cpu" # gpu not fully supported yet
    wakeword_model: str = realtime_audio.DEFAULT_WAKEWORD_MODEL

def main(
    args: Args
):

    audio_gen = AudioGeneratorModel(args.audio_generator_checkpoint_path, args.audio_generator_config_path, device=args.device)
    llm = LLM(model_name=args.llm, ollama_api_url=args.ollama_api_url)

    llm_audio_gen = LLMAudioGenerator(audio_gen, llm)
    rt = RealtimeTranscription(
        transcription_model=args.transcription_model,
        buffer_duration=args.buffer_duration,
        device=args.device,
        wakeword_model=args.wakeword_model
    )
    rt.register_callback(llm_audio_gen.query)
    
    # run on audio
    rt()


if __name__ == "__main__":
    main()