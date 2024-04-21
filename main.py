import boto3
import json
import sys


brt = boto3.client(service_name='bedrock-runtime')

# body = json.dumps({
#     'prompt': '\n\nHuman: write an essay for living on mars in 1000 words\n\nAssistant:',
#     'max_tokens_to_sample': 4000
# })


body = json.dumps({
    "prompt": '\n\nHuman: write an essay for living on mars in 100 words\n\nAssistant:',
    "max_gen_len": 512,
    "temperature": 0.1,
    "top_p": 0.9
})

    #modelId='anthropic.claude-v2', 
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
#
# response = brt.invoke_model(modelId='meta.llama2-13b-chat-v1', body=body)
# print(response)
# response_body = json.loads(response.get('body').read())
# print(response_body)