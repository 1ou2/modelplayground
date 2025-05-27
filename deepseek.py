import boto3
from botocore.exceptions import ClientError
import json
# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g. DeepSeek-R1
model_id = "us.deepseek.r1-v1:0"
#model_id = "anthropic.claude-3-haiku-20240307-v1:0"
#model_id = "mistral.mistral-large-2402-v1:0"



prompt = "Explique moi qui est Donald Trump."

# Configurer les paramètres pour DeepSeek
request_body = {
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
}

# Invoquer le modèle DeepSeek
response = client.invoke_model(
    modelId="us.deepseek.r1-v1:0",  # Vérifiez l'ID exact du modèle
    body=json.dumps(request_body)
)
# Traiter la réponse
response_body = json.loads(response["body"].read())
print(response_body)

generated_text = response_body.get("generation", "")
print(generated_text)

# Start a conversation with a user message and the document
conversation = [
    {
        "role": "user",
        "content": [
            {"text": "Briefly explain the main technical components of a LLM"}
        ],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.3},
    )

    # Extract and print the reasoning and response text.
    reasoning, response_text = "", ""
    for item in response["output"]["message"]["content"]:
        for key, value in item.items():
            if key == "reasoningContent":
                reasoning = value["reasoningText"]["text"]
            elif key == "text":
                response_text = value

    print(f"\nReasoning:\n{reasoning}")
    print(f"\nResponse:\n{response_text}")

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)