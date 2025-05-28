# Amazon Bedrock Models Integration

This project integrates Amazon Bedrock models (DeepSeek, Claude, Mistral) into a Flask web application.

## Setup

1. Make sure you have the necessary AWS permissions for Amazon Bedrock.
2. Configure your AWS credentials using AWS CLI or environment variables.
Create a dedicated user in AWS.
Add bedrock policy
Create access key
Then copy access key in ¯/.aws/credentials
```ini
[default]
aws_access_key_id=ABCDEFG
aws_secret_access_key=MY_SECRET
```

3. Install the required packages:
   ```
   pip install flask boto3
   ```

## Files

- `app.py` - Flask web application
- `bedrock_models.py` - Modular implementation of different Bedrock models
- `model_config.json` - Configuration for available models
- `test_models.py` - Script to test models individually

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Test models individually:
   ```
   python test_models.py
   ```

## Switching Models

The application supports switching between different models:
- DeepSeek
- Claude
- Mistral

You can configure which models are available in the `model_config.json` file.

# Google

## Install Google CLI
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli

## Launch init
```gcloud init --console-only```
Copy url in browser
Authenticate with corporate account
Copy authorization code back in shell.

Result
```
Your project default Compute Engine zone has been set to [europe-west1-c].
Your project default Compute Engine region has been set to [europe-west1].
You can change it by running [gcloud config set compute/region NAME].
Created a default .boto configuration file at [/home/gabriel/.boto]. See this file and
[https://cloud.google.com/storage/docs/gsutil/commands/config] for more
information about configuring Google Cloud Storage.
```

Configurations are stored in your user config directory (typically ```~/.config/gcloud```)
```gcloud config configurations list```

Authenticate 
```gcloud auth application-default login```
A message is displayed, click on link
Your browser has been opened to visit: 
    https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=

As a result a credential file is created
```
more ~/.config/gcloud/application_default_credentials.json 
{
  "account": "",
  "client_id": "XXXX.apps.googleusercontent.com",
  "client_secret": "XXXX",
  "quota_project_id": "XXXX",
  "refresh_token": "XXXX",
  "type": "authorized_user",
  "universe_domain": "googleapis.com"
}
```