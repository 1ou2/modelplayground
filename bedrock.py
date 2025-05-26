import boto3

import json


region = 'us-east-1'
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=region
)


def list_foundation_models():
    response = bedrock_client.list_foundation_models()

    return response['modelSummaries']


def get_foundation_model(model_id):
    response = bedrock_client.get_foundation_model(
        modelIdentifier=model_id
    )

    return response['modelDetails']

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
)

def invoke(prompt, temperature=1.0, max_tokens=500):
    prompt_config = {
        "prompt": f'\n\nHuman: {prompt} \n\nAssistant:',
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(prompt_config),
        modelId="anthropic.claude-v2"
    )

    response_body = json.loads(response.get("body").read())

    return response_body.get("completion")

if __name__ == "__main__":
    print(list_foundation_models())
    model_id = "anthropic.claude-v2"
    print(get_foundation_model(model_id))
    prompt = "Tell me a short joke"
    print(invoke(prompt))
