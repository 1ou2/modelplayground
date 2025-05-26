# app.py
from flask import Flask, request, jsonify, render_template
import json
import os
import boto3
import json

app = Flask(__name__)

# Load model keys
with open('model_keys.json', 'r') as f:
    model_keys = json.load(f)

# Load or initialize conversations
CONVERSATIONS_FILE = 'conversations.json'
if os.path.exists(CONVERSATIONS_FILE):
    with open(CONVERSATIONS_FILE, 'r') as f:
        conversations = json.load(f)
else:
    conversations = {}

# Save conversations
def save_conversations():
    with open(CONVERSATIONS_FILE, 'w') as f:
        json.dump(conversations, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    data = request.json
    conversation_id = data.get('conversation_id')
    message = data.get('message')
    model = data.get('model')

    # Initialize conversation if new
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    # Append user message
    conversations[conversation_id].append({'role': 'user', 'content': message})

    # Call API with model and corresponding key
    response = call_llm_api(conversations[conversation_id], model)

    # Append assistant response
    conversations[conversation_id].append({'role': 'assistant', 'content': response})

    save_conversations()

    return jsonify({'response': response})

@app.route('/model_keys')
def get_model_keys():
    with open('model_keys.json', 'r') as f:
        return jsonify(json.load(f))


def call_llm_api(messages, model):
    api_key = model_keys.get(model)
    if not api_key:
        return "Error: API key for model not found."
    # Call your API with the selected api_key

    # return echo messages
    return messages[-1]['content'] 

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        data = request.json
        # Save selected model
        # For simplicity, store in a file or variable
        with open('settings.json', 'w') as f:
            json.dump(data, f)
        return jsonify({'status': 'saved'})
    else:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        else:
            settings = {'model': 'default-model'}
        return jsonify(settings)

if __name__ == '__main__':
    app.run(debug=True)
