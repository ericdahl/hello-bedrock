#!/usr/bin/env uv run
# /// script
# dependencies = ["boto3"]
# ///
import boto3
import json
import sys


session = boto3.Session()
brt = session.client(service_name='bedrock-runtime')

def prompt_titan_text(prompt: str):

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1
        }
    })

    response = brt.invoke_model_with_response_stream(
        modelId='amazon.titan-text-lite-v1',
        body=body
    )
    print(response)
    for e in response['body']:
        out_tokens = json.loads(e['chunk']['bytes'])['outputText']
        print(out_tokens)

def prompt_llama(prompt: str):
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.1,
        "top_p": 0.9
    })
    response = brt.invoke_model_with_response_stream(
        modelId='meta.llama2-13b-chat-v1',
        body=body
    )

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                d = json.loads(chunk.get('bytes').decode())
                print(d['generation'], end='')
                sys.stdout.flush()

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
def prompt_cohere(prompt: str):
    body = json.dumps({
        "prompt": prompt,
        "stream": True
    })
    response = brt.invoke_model_with_response_stream(
        modelId="cohere.command-light-text-v14",
        body=body
    )

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                d = json.loads(chunk.get('bytes').decode())
                if not d["is_finished"]:
                    print(d['text'], end='')
                    sys.stdout.flush()

def prompt_claude(prompt: str):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    response = brt.invoke_model_with_response_stream(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=body
    )

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                d = json.loads(chunk.get('bytes').decode())
                if d.get('type') == 'content_block_delta':
                    if 'text' in d.get('delta', {}):
                        print(d['delta']['text'], end='')
                        sys.stdout.flush()


prompt = "Generate a 7 day itinerary for a vacation to Japan in June. Interests include experiencing modern " \
         "culture, unusual sights, immersion."
# prompt_llama(prompt)
# prompt_titan_text(prompt)
# prompt_cohere(prompt)
prompt_claude(prompt)