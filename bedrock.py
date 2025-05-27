import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

region = 'us-east-1'
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=region
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
)

def list_foundation_models():
    response = bedrock_client.list_foundation_models()

    return response['modelSummaries']


def get_foundation_model(model_id):
    response = bedrock_client.get_foundation_model(
        modelIdentifier=model_id
    )

    return response['modelDetails']




def list_inference_profiles():
    """List all available inference profiles"""
    response = bedrock_client.list_inference_profiles()
    return response.get('inferenceProfileSummaries', [])

def stream_response(user_message,model_id="mistral.mistral-large-2402-v1:0"):
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        streaming_response = bedrock_runtime.converse_stream(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )

        # Extract and print the streamed response text in real-time.
        for chunk in streaming_response["stream"]:
            if "contentBlockDelta" in chunk:
                text = chunk["contentBlockDelta"]["delta"]["text"]
                print(text, end="")
        print()  # Print a newline at the end of the response

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

if __name__ == "__main__":
    #print(list_foundation_models())
    all_models = list_foundation_models()
    for model in all_models:
        print(f"{model['modelName']} : {model['modelId']}")
    
    # Get details for Claude model
    model_id = "anthropic.claude-v2"
    print(get_foundation_model(model_id))
    
    # For DeepSeek model, we need to use an inference profile
    # First, check if you have any inference profiles available
    profiles = list_inference_profiles()
    print("Available inference profiles:")
    for profile in profiles:
        print(f"Profile: {profile.get('inferenceProfileName')} - ID: {profile.get('inferenceProfileId')}")
    
    # Try both Claude and DeepSeek models
    prompt = "Explain what are the main components of a LLM"
    prompt = "Briefly explain the main technical components of a LLM"
    stream_response(prompt)
    