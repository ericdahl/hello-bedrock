#!/usr/bin/env uv run
# /// script
# dependencies = ["boto3"]
# ///
import boto3
import json
import sys
import argparse


session = boto3.Session()
brt = session.client(service_name='bedrock-runtime')

def prompt_titan(prompt: str):
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    })

    try:
        response = brt.invoke_model_with_response_stream(
            modelId='amazon.titan-tg1-large',
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if 'outputText' in d:
                        print(d['outputText'], end='')
                        sys.stdout.flush()
    except Exception as e:
        print(f"Error calling Titan: {e}", file=sys.stderr)
        raise

def prompt_llama(prompt: str):
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId='us.meta.llama3-1-70b-instruct-v1:0',
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if 'generation' in d:
                        print(d['generation'], end='')
                        sys.stdout.flush()
    except Exception as e:
        print(f"Error calling Llama: {e}", file=sys.stderr)
        raise

def prompt_cohere(prompt: str):
    body = json.dumps({
        "message": prompt,
        "max_tokens": 4096,
        "temperature": 0.7
    })
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId="cohere.command-r-v1:0",
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if d.get('event_type') == 'text-generation':
                        if 'text' in d:
                            print(d['text'], end='')
                            sys.stdout.flush()
                    elif d.get('event_type') == 'stream-end':
                        break
    except Exception as e:
        print(f"Error calling Cohere: {e}", file=sys.stderr)
        raise

def prompt_claude(prompt: str):
    """Use Anthropic Claude Haiku 4.5 model"""
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
    
    try:
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
    except Exception as e:
        print(f"Error calling Claude: {e}", file=sys.stderr)
        raise

def prompt_mistral(prompt: str):
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId="mistral.mixtral-8x7b-instruct-v0:1",
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    # Mistral streaming response format
                    if 'outputs' in d:
                        for output in d['outputs']:
                            if 'text' in output:
                                print(output['text'], end='')
                                sys.stdout.flush()
                    elif 'text' in d:
                        print(d['text'], end='')
                        sys.stdout.flush()
                    elif 'chunk' in d and 'text' in d['chunk']:
                        print(d['chunk']['text'], end='')
                        sys.stdout.flush()
    except Exception as e:
        print(f"Error calling Mistral: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description='Generate text using AWS Bedrock models')
    parser.add_argument('--model', '-m', 
                       choices=['claude', 'titan', 'llama', 'cohere', 'mistral'],
                       default='claude',
                       help='Model to use (default: claude)')
    parser.add_argument('prompt', 
                       nargs='?',
                       default="Generate a 7 day itinerary for a vacation to Japan in June. Interests include experiencing modern culture, unusual sights, immersion.",
                       help='Prompt to send to the model')
    
    args = parser.parse_args()
    
    model_functions = {
        'claude': prompt_claude,
        'titan': prompt_titan,
        'llama': prompt_llama,
        'cohere': prompt_cohere,
        'mistral': prompt_mistral
    }
    
    model_func = model_functions[args.model]
    model_func(args.prompt)
    print()


if __name__ == '__main__':
    main()
