# app.py
from flask import Flask, request, jsonify, render_template
import json
import os
import boto3
import json

app = Flask(__name__)

# Load model configuration
MODEL_CONFIG_FILE = 'model_config.json'
if os.path.exists(MODEL_CONFIG_FILE):
    with open(MODEL_CONFIG_FILE, 'r') as f:
        model_config = json.load(f)
else:
    # Default configuration
    model_config = {
        "available_models": ["deepseek", "claude", "mistral"],
        "default_model": "deepseek",
        "region": "us-east-1"
    }
    with open(MODEL_CONFIG_FILE, 'w') as f:
        json.dump(model_config, f)

# Load or initialize conversations
CONVERSATIONS_FILE = 'conversations.json'
if os.path.exists(CONVERSATIONS_FILE):
    with open(CONVERSATIONS_FILE, 'r') as f:
        conversations = json.load(f)
else:
    conversations = {
        "metadata": {},  # Store conversation titles and timestamps
        "data": {}       # Store actual conversation messages
    }

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
    if conversation_id not in conversations["data"]:
        conversations["data"][conversation_id] = []
        # Create metadata entry with timestamp as default title
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversations["metadata"][conversation_id] = {
            "title": timestamp,
            "created_at": timestamp,
            "updated_at": timestamp,
            "model": model
        }

    # Append user message
    conversations["data"][conversation_id].append({'role': 'user', 'content': message})

    # Call API with model and corresponding key
    result = call_llm_api(conversations["data"][conversation_id], model)
    
    # Extract response and reasoning
    response = result.get('response', '')
    reasoning = result.get('reasoning', '')

    # Append assistant response (store only the final response in conversation history)
    conversations["data"][conversation_id].append({'role': 'assistant', 'content': response})
    
    # Update the timestamp
    import datetime
    conversations["metadata"][conversation_id]["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_conversations()

    # Return both response and reasoning to the frontend
    return jsonify({
        'response': response,
        'reasoning': reasoning,
        'has_reasoning': bool(reasoning),
        'conversation_id': conversation_id
    })

@app.route('/models')
def get_models():
    return jsonify({
        "available_models": model_config["available_models"],
        "default_model": model_config["default_model"]
    })

@app.route('/conversations', methods=['GET'])
def get_conversations():
    """Get all conversation metadata"""
    return jsonify(conversations["metadata"])

@app.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get a specific conversation"""
    if conversation_id in conversations["data"]:
        return jsonify({
            "messages": conversations["data"][conversation_id],
            "metadata": conversations["metadata"].get(conversation_id, {})
        })
    return jsonify({"error": "Conversation not found"}), 404

@app.route('/conversations/<conversation_id>/title', methods=['PUT'])
def update_conversation_title(conversation_id):
    """Update a conversation title"""
    data = request.json
    new_title = data.get('title')
    
    if conversation_id in conversations["metadata"] and new_title:
        conversations["metadata"][conversation_id]["title"] = new_title
        save_conversations()
        return jsonify({"status": "success"})
    
    return jsonify({"error": "Conversation not found or invalid title"}), 400

@app.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation"""
    if conversation_id in conversations["data"]:
        # Remove the conversation data
        del conversations["data"][conversation_id]
        # Remove the metadata
        if conversation_id in conversations["metadata"]:
            del conversations["metadata"][conversation_id]
        save_conversations()
        return jsonify({"status": "success"})
    
    return jsonify({"error": "Conversation not found"}), 404


def call_llm_api(messages, model_name):
    """
    Call the appropriate LLM API based on the model name.
    
    Parameters:
        messages (list): List of message dictionaries with 'role' and 'content'
        model_name (str): Name of the model to use
        
    Returns:
        dict: Contains 'response' and optionally 'reasoning' if available
    """
    try:
        from bedrock_models import get_model
        
        # Get the appropriate model instance
        model = get_model(model_name)
        

        response = model.generate(messages[-1]['content'])
        result = {"response": response, "reasoning": ""}
            
        # Return both response and reasoning if available
        return result

    except Exception as e:
        return {"response": f"Error: {str(e)}", "reasoning": ""}

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
