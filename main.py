import boto3
import json
import sys


brt = boto3.client(service_name='bedrock-runtime')

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
        "prompt": prompt,
        # "stream": True,
        "max_tokens_to_sample": 4000
    })
    response = brt.invoke_model_with_response_stream(
        modelId="anthropic.claude-instant-v1",
        body=body
    )

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                d = json.loads(chunk.get('bytes').decode())
                print(d['completion'], end='')
                sys.stdout.flush()


prompt = "Human: Generate a 7 day itinerary for a vacation to Japan in June. Interests include experiencing modern " \
         "culture, unusual sights, immersion.  Assistant:"
# prompt_llama(prompt)
# prompt_titan_text(prompt)
# prompt_cohere(prompt)
prompt_claude(prompt)