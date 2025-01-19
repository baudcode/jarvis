from src.llm_audio_generator import main, Args
from src import llm, audio_generator, realtime_audio
import dataclasses
import argparse



def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audio_generator_checkpoint_path", default=audio_generator.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--audio_generator_config_path", default=audio_generator.DEFAULT_CONFIG_PATH)
    parser.add_argument("--ollama_api_url", default=llm.OLLAMA_API_URL)
    parser.add_argument("--llm", default=llm.DEFAULT_MODEL)
    parser.add_argument("--transcription_model", default=realtime_audio.DEFAULT_TRANSCRIPTION_MODEL)
    parser.add_argument("--wakeword_model", default=realtime_audio.DEFAULT_WAKEWORD_MODEL)
    parser.add_argument("--buffer_duration", type=int, default=realtime_audio.DEFAULT_BUFFER_DURATION)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    return Args(**dict(args.__dict__))
    


if __name__ == "__main__":
    main(
        get_args()
    )