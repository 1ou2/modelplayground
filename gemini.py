import os
from dotenv import load_dotenv


load_dotenv()
GOOGLE_CLOUD_PROJECT= os.getenv("GOOGLE_CLOUD_PROJECT", "modelplayground-416920")
GOOGLE_CLOUD_LOCATION=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

if GOOGLE_GENAI_USE_VERTEXAI == "True":
    GOOGLE_GENAI_USE_VERTEXAI = True

print(f"GOOGLE_CLOUD_PROJECT: {GOOGLE_CLOUD_PROJECT}")
print(f"GOOGLE_CLOUD_LOCATION: {GOOGLE_CLOUD_LOCATION}")
print(f"GOOGLE_GENAI_USE_VERTEXAI: {GOOGLE_GENAI_USE_VERTEXAI}")

if GOOGLE_GENAI_USE_VERTEXAI:
    print("Using Vertex AI")

from google import genai
from google.genai.types import HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="How does AI work?",
)
print(response.text)