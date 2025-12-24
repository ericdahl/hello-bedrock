#!/usr/bin/env uv run
# /// script
# dependencies = ["boto3"]
# ///
import boto3
import json
import sys
import argparse
from typing import Dict, Tuple, Optional

# Pricing per 1M tokens (input, output) - approximate pricing as of 2025
# Prices are in USD per 1M tokens
PRICING = {
    'claude': {
        'model_id': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'input_price': 0.25,   # $0.25 per 1M input tokens
        'output_price': 1.25   # $1.25 per 1M output tokens
    },
    'titan': {
        'model_id': 'amazon.titan-tg1-large',
        'input_price': 0.8,    # $0.80 per 1M input tokens
        'output_price': 0.8    # $0.80 per 1M output tokens
    },
    'llama': {
        'model_id': 'us.meta.llama3-1-70b-instruct-v1:0',
        'input_price': 0.65,   # $0.65 per 1M input tokens
        'output_price': 0.65   # $0.65 per 1M output tokens
    },
    'cohere': {
        'model_id': 'cohere.command-r-v1:0',
        'input_price': 0.5,    # $0.50 per 1M input tokens
        'output_price': 1.5    # $1.50 per 1M output tokens
    },
    'mistral': {
        'model_id': 'mistral.mixtral-8x7b-instruct-v0:1',
        'input_price': 0.27,   # $0.27 per 1M input tokens
        'output_price': 0.27   # $0.27 per 1M output tokens
    }
}

session = boto3.Session()
brt = session.client(service_name='bedrock-runtime')

def count_tokens(model_id: str, body: str) -> int:
    """Count input tokens using Bedrock CountTokens API"""
    try:
        response = brt.count_tokens(modelId=model_id, body=body)
        return response.get('totalTokens', 0)
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(body) // 4

def prompt_titan(prompt: str) -> Dict:
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    })
    
    model_id = PRICING['titan']['model_id']
    input_tokens = count_tokens(model_id, body)
    output_tokens = 0
    output_text = ""

    try:
        response = brt.invoke_model_with_response_stream(
            modelId=model_id,
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if 'outputText' in d:
                        text = d['outputText']
                        output_text += text
                        print(text, end='')
                        sys.stdout.flush()
                    # Check for usage metadata
                    if 'usage' in d:
                        output_tokens = d['usage'].get('outputTokens', 0)
    except Exception as e:
        print(f"Error calling Titan: {e}", file=sys.stderr)
        raise
    
    # Estimate output tokens if not provided
    if output_tokens == 0:
        output_tokens = len(output_text) // 4
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_id': model_id
    }

def prompt_llama(prompt: str) -> Dict:
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    model_id = PRICING['llama']['model_id']
    input_tokens = count_tokens(model_id, body)
    output_tokens = 0
    output_text = ""
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId=model_id,
            body=body
        )

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if 'generation' in d:
                        text = d['generation']
                        output_text += text
                        print(text, end='')
                        sys.stdout.flush()
                    # Check for usage metadata
                    if 'usage' in d:
                        output_tokens = d['usage'].get('outputTokens', 0)
    except Exception as e:
        print(f"Error calling Llama: {e}", file=sys.stderr)
        raise
    
    # Estimate output tokens if not provided
    if output_tokens == 0:
        output_tokens = len(output_text) // 4
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_id': model_id
    }

def prompt_cohere(prompt: str) -> Dict:
    body = json.dumps({
        "message": prompt,
        "max_tokens": 4096,
        "temperature": 0.7
    })
    
    model_id = PRICING['cohere']['model_id']
    input_tokens = count_tokens(model_id, body)
    output_tokens = 0
    output_text = ""
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId=model_id,
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
                            text = d['text']
                            output_text += text
                            print(text, end='')
                            sys.stdout.flush()
                    elif d.get('event_type') == 'stream-end':
                        if 'usage' in d:
                            output_tokens = d['usage'].get('output_tokens', 0)
                        break
    except Exception as e:
        print(f"Error calling Cohere: {e}", file=sys.stderr)
        raise
    
    # Estimate output tokens if not provided
    if output_tokens == 0:
        output_tokens = len(output_text) // 4
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_id': model_id
    }

def prompt_claude(prompt: str) -> Dict:
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
    
    model_id = PRICING['claude']['model_id']
    input_tokens = count_tokens(model_id, body)
    output_tokens = 0
    output_text = ""
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId=model_id,
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
                            text = d['delta']['text']
                            output_text += text
                            print(text, end='')
                            sys.stdout.flush()
                    elif d.get('type') == 'message_stop':
                        if 'usage' in d:
                            output_tokens = d['usage'].get('output_tokens', 0)
    except Exception as e:
        print(f"Error calling Claude: {e}", file=sys.stderr)
        raise
    
    # Estimate output tokens if not provided
    if output_tokens == 0:
        output_tokens = len(output_text) // 4
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_id': model_id
    }

def prompt_mistral(prompt: str) -> Dict:
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    model_id = PRICING['mistral']['model_id']
    input_tokens = count_tokens(model_id, body)
    output_tokens = 0
    output_text = ""
    
    try:
        response = brt.invoke_model_with_response_stream(
            modelId=model_id,
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
                                text = output['text']
                                output_text += text
                                print(text, end='')
                                sys.stdout.flush()
                    elif 'text' in d:
                        text = d['text']
                        output_text += text
                        print(text, end='')
                        sys.stdout.flush()
                    elif 'chunk' in d and 'text' in d['chunk']:
                        text = d['chunk']['text']
                        output_text += text
                        print(text, end='')
                        sys.stdout.flush()
                    # Check for usage metadata
                    if 'usage' in d:
                        output_tokens = d['usage'].get('output_tokens', 0)
    except Exception as e:
        print(f"Error calling Mistral: {e}", file=sys.stderr)
        raise
    
    # Estimate output tokens if not provided
    if output_tokens == 0:
        output_tokens = len(output_text) // 4
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model_id': model_id
    }

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
    """Calculate costs for input, output, and total"""
    pricing = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing['input_price']
    output_cost = (output_tokens / 1_000_000) * pricing['output_price']
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

def print_diagnostics(model: str, diagnostics: Dict):
    """Print token usage and cost diagnostics"""
    input_tokens = diagnostics['input_tokens']
    output_tokens = diagnostics['output_tokens']
    total_tokens = input_tokens + output_tokens
    model_id = diagnostics['model_id']
    
    input_cost, output_cost, total_cost = calculate_cost(model, input_tokens, output_tokens)
    
    print("\n" + "="*60, file=sys.stderr)
    print("DIAGNOSTICS", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Model: {model_id}", file=sys.stderr)
    print(f"Input tokens:  {input_tokens:,}", file=sys.stderr)
    print(f"Output tokens: {output_tokens:,}", file=sys.stderr)
    print(f"Total tokens:  {total_tokens:,}", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    print(f"Input cost:  ${input_cost:.6f}", file=sys.stderr)
    print(f"Output cost: ${output_cost:.6f}", file=sys.stderr)
    print(f"Total cost:  ${total_cost:.6f}", file=sys.stderr)
    print("="*60, file=sys.stderr)

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
    diagnostics = model_func(args.prompt)
    print()
    print_diagnostics(args.model, diagnostics)


if __name__ == '__main__':
    main()
