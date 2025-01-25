import requests

# OpenAI API standard endpoint
SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"

request_data = {
    "messages": [
        {"role": "user", "content": "Tell me a story about a girl and a boy?"}
    ], 
    "max_completion_tokens": 32, 
}

if __name__ == "__main__":
    response = requests.post(SERVER_URL, stream=False, json=request_data)    
    print(response.json())
