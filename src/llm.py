import requests
import json
import traceback

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:1b"

def query_model(question: str, model_name: str = DEFAULT_MODEL, ollama_api_url: str = OLLAMA_API_URL, debug=False):
  
    # Construct the request payload
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": question
        }],
    }

    # Send the request to the Ollama API
    try:
        with requests.post(ollama_api_url, json=payload, stream=True, headers={
            "Content-Type": "application/json"
        }) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    content

                data = json.loads(line.decode())
                content = data['message']['content']
                if debug:
                    print(content, end="", flush=True)            
                yield content
            
            # return total_response

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
        print(traceback.format_exc())

def sentence_iterator(generator):
    total_response = ""
    prev = 0

    for r in generator:
        start = len(total_response)  
        total_response += r

        for j, c in enumerate(r):
            i = j + start
            if c == '.' and not total_response[i-1].isnumeric():
                sentence = total_response[prev:i+1].strip()
                yield sentence
                prev = i + 1


def test():

    generator = query_model(
        "Whats the size of the universe?. Give an answer in no more than five sentences.", "llama3.2:1b"
    )
    
    for sentence in sentence_iterator(generator):
        print(sentence)

def test_generate_wavs():
    from audio_generator import AudioGeneratorModel, play
    
    model = AudioGeneratorModel()
    generator = query_model(
        "Whats the size of the universe?. Give an answer in no more than five sentences.", "llama3.2:1b"
    )
    for sentence in sentence_iterator(generator):
        audio_raw = model(sentence)
        play(audio_raw, model.rate)

class LLM:
    def __init__(self, model_name: str = DEFAULT_MODEL, ollama_api_url: str = OLLAMA_API_URL, debug=False):
        self.model_name = model_name
        self.ollama_api_url = ollama_api_url
        self.debug = debug

    
    def query(self, question: str):
        return query_model(question, model_name=self.model_name, ollama_api_url=self.ollama_api_url, debug=self.debug)

    def query_sentences(self, question: str):
        return sentence_iterator(self.query(question))
        


if __name__ == "__main__":
    test_generate_wavs()
