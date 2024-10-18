import boto3
import json
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_bedrock_model (client, prompt, modelId):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    )
    
    accept = 'application/json'
    contentType = 'application/json'
    response = client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]["text"], response_body['usage']['input_tokens'], response_body['usage']['output_tokens']

def query_bedrock_model_with_image(client, prompt, photo_data, modelId):
    photo_base64 = base64.b64encode(photo_data).decode('utf-8')
    body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": photo_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                    ],
                }
            ],
    }
    )

    response = client.invoke_model(
        modelId=modelId,
        body=body
    )
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]["text"], response_body['usage']['input_tokens'], response_body['usage']['output_tokens']

def encode_uploaded_file(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def query_bedrock_with_multiple_images(bedrock, prompt, uploaded_files, modelId):


    # Encode all uploaded images
    images = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": uploaded_file.type,
                "data": encode_uploaded_file(uploaded_file)
            }
        } 
        for uploaded_file in uploaded_files
    ]

    # Prepare the request body
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "messages": [
            {
                "role": "user",
                "content": images + [{"type": "text", "text": prompt}]
            }
        ]
    })
    counter = time.time()
    # Invoke the model
    response = bedrock.invoke_model(
        modelId=modelId,
        body=body
    )
    print (time.time()-counter)
    
    # Parse and return the response
    response_body = json.loads(response['body'].read())
    
    return (response_body['content'][0]["text"],
                response_body['usage']['input_tokens'],
                response_body['usage']['output_tokens'])
