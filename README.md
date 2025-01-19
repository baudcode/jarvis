# Hey Jarvis - Realtime LLM Buddy


LLM to talk to in real time.
It uses:

- llm: [ollama](https://ollama.com/) (default: [llama3.2:1b](https://ollama.com/library/llama3.2:1b))
- audio to text: [whisperX](https://github.com/m-bain/whisperX)
- text to audio: [VITS](https://github.com/jaywalnut310/vits/tree/main)
- wakeword: [openWakeWord](https://github.com/dscripka/openWakeWord) (hey jarvis)

### Setup

- Fast setup script:
```bash
bash setup.sh
```

### Run

```bash
python main.py
```