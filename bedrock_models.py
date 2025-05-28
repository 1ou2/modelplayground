import boto3
from botocore.exceptions import ClientError
import json

class BedrockModel:
    """
    Base class for Amazon Bedrock models.
    
    This class provides the foundation for specific model implementations
    and handles common functionality like client initialization.
    """
    
    def __init__(self, model_id, region_name="us-east-1"):
        """
        Initialize a Bedrock model.
        
        Parameters:
            model_id (str): The ID of the model to use
            region_name (str): AWS region where the model is available
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text based on a single prompt.
        
        Parameters:
            prompt (str): The input text to generate from
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            
        Returns:
            str: The generated text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def converse(self, messages, max_tokens=2000, temperature=0.3):
        """
        Have a conversation with the model using multiple messages.
        
        Parameters:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            
        Returns:
            dict: Contains 'response' (str) and optionally 'reasoning' (str)
        """
        raise NotImplementedError("Subclasses must implement this method")


class DeepSeekModel(BedrockModel):
    """
    Implementation of the DeepSeek model for Amazon Bedrock.
    
    This class handles the specific request/response formats for DeepSeek models.
    """
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text using the DeepSeek model.
        
        Parameters:
            prompt (str): The input text to generate from
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            
        Returns:
            str: The generated text
        """
        request_body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response["body"].read())
            return response_body["choices"][0]["text"]
        except (ClientError, Exception) as e:
            return f"Error: {str(e)}"
    
    def converse(self, messages, max_tokens=2000, temperature=0.3):
        """
        Have a conversation with the DeepSeek model.
        
        Parameters:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            
        Returns:
            dict: Contains 'response' (str) and 'reasoning' (str) if available
        """
        formatted_messages = []
        
        for message in messages:
            if isinstance(message["content"], str):
                formatted_message = {
                    "role": message["role"],
                    "content": [{"text": message["content"]}]
                }
            else:
                formatted_message = message
            formatted_messages.append(formatted_message)
        
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=formatted_messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
            )
            
            response_text = ""
            reasoning = ""
            
            for item in response["output"]["message"]["content"]:
                for key, value in item.items():
                    if key == "reasoningContent":
                        reasoning = value["reasoningText"]["text"]
                    elif key == "text":
                        response_text = value
            
            return {
                "response": response_text,
                "reasoning": reasoning
            }
        except (ClientError, Exception) as e:
            return {"response": f"Error: {str(e)}", "reasoning": ""}


class ClaudeModel(BedrockModel):
    """
    Implementation of the Claude model for Amazon Bedrock.
    
    This class handles the specific request/response formats for Claude models.
    """
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text using the Claude model.
        
        Parameters:
            prompt (str): The input text to generate from
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            
        Returns:
            str: The generated text
        """
        request_body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("completion", "")
        except (ClientError, Exception) as e:
            return f"Error: {str(e)}"
    
    def converse(self, messages, max_tokens=2000, temperature=0.3):
        """
        Have a conversation with the Claude model.
        
        Parameters:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            
        Returns:
            dict: Contains 'response' (str) and 'reasoning' (str) if available
        """
        formatted_messages = []
        
        for message in messages:
            if isinstance(message["content"], str):
                formatted_message = {
                    "role": message["role"],
                    "content": [{"text": message["content"]}]
                }
            else:
                formatted_message = message
            formatted_messages.append(formatted_message)
        
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=formatted_messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
            )
            
            response_text = ""
            reasoning = ""
            
            for item in response["output"]["message"]["content"]:
                if "text" in item:
                    response_text = item["text"]
                # Check if Claude provides reasoning (uncommon but possible)
                if "reasoningContent" in item:
                    reasoning = item["reasoningContent"]["reasoningText"]["text"]
            
            return {
                "response": response_text,
                "reasoning": reasoning
            }
        except (ClientError, Exception) as e:
            return {"response": f"Error: {str(e)}", "reasoning": ""}


class MistralModel(BedrockModel):
    """
    Implementation of the Mistral model for Amazon Bedrock.
    
    This class handles the specific request/response formats for Mistral models.
    """
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text using the Mistral model.
        
        Parameters:
            prompt (str): The input text to generate from
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            
        Returns:
            str: The generated text
        """
        request_body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("outputs", [{}])[0].get("text", "")
        except (ClientError, Exception) as e:
            return f"Error: {str(e)}"
    
    def converse(self, messages, max_tokens=2000, temperature=0.3):
        """
        Have a conversation with the Mistral model.
        
        Parameters:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            
        Returns:
            dict: Contains 'response' (str) and 'reasoning' (str) if available
        """
        formatted_messages = []
        
        for message in messages:
            if isinstance(message["content"], str):
                formatted_message = {
                    "role": message["role"],
                    "content": [{"text": message["content"]}]
                }
            else:
                formatted_message = message
            formatted_messages.append(formatted_message)
        
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=formatted_messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
            )
            
            response_text = ""
            reasoning = ""
            
            for item in response["output"]["message"]["content"]:
                if "text" in item:
                    response_text = item["text"]
                # Check if Mistral provides reasoning
                if "reasoningContent" in item:
                    reasoning = item["reasoningContent"]["reasoningText"]["text"]
            
            return {
                "response": response_text,
                "reasoning": reasoning
            }
        except (ClientError, Exception) as e:
            return {"response": f"Error: {str(e)}", "reasoning": ""}


def get_model(model_name, region_name="us-east-1"):
    """
    Factory function to create the appropriate model instance.
    
    This function creates and returns an instance of the requested model class
    based on the model name provided.
    
    Parameters:
        model_name (str): Name of the model to create ('deepseek', 'claude', or 'mistral')
        region_name (str): AWS region where the model is available
        
    Returns:
        BedrockModel: An instance of the appropriate model class
        
    Raises:
        ValueError: If the model name is not recognized
    """
    model_map = {
        "deepseek": {
            "class": DeepSeekModel,
            "id": "us.deepseek.r1-v1:0"
        },
        "claude": {
            "class": ClaudeModel,
            "id": "anthropic.claude-3-haiku-20240307-v1:0"
        },
        "mistral": {
            "class": MistralModel,
            "id": "mistral.mistral-large-2402-v1:0"
        }
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = model_map[model_name]
    return model_info["class"](model_info["id"], region_name)