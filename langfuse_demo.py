"""
LangFuse Integration Demo for AWS Bedrock

This demo shows how to use LangFuse (https://langfuse.com/) for observability
and tracing of AWS Bedrock LLM calls.

LangFuse provides:
- Trace tracking for LLM calls
- Cost monitoring
- Latency tracking
- User feedback collection
- Prompt version management
"""

import boto3
import json
import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langfuse import Langfuse

# Load environment variables from .env file
load_dotenv()

# Initialize LangFuse client
# Get credentials from environment variables
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Use cloud by default
)

# Initialize AWS Bedrock client
brt = boto3.client(service_name='bedrock-runtime')


def prompt_claude_with_langfuse(prompt: str, user_id: str = "demo-user"):
    """
    Demonstrate Claude invocation with LangFuse tracing.

    Args:
        prompt: The prompt to send to Claude
        user_id: Identifier for the user making the request
    """
    # Create a trace in LangFuse
    trace = langfuse.trace(
        name="bedrock-claude-call",
        user_id=user_id,
        metadata={
            "model": "anthropic.claude-instant-v1",
            "provider": "aws-bedrock"
        }
    )

    # Create a generation span within the trace
    generation = trace.generation(
        name="claude-generation",
        model="anthropic.claude-instant-v1",
        input=prompt,
        metadata={
            "temperature": 0,
            "max_tokens": 4000
        }
    )

    start_time = time.time()

    try:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 4000
        })

        response = brt.invoke_model_with_response_stream(
            modelId="anthropic.claude-instant-v1",
            body=body
        )

        # Collect the full response
        full_response = ""
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    chunk_text = d.get('completion', '')
                    full_response += chunk_text
                    print(chunk_text, end='')
                    sys.stdout.flush()

        print()  # New line after response

        # Calculate duration
        duration = time.time() - start_time

        # Update the generation with the output and metrics
        generation.end(
            output=full_response,
            metadata={
                "latency_seconds": duration,
                "completion_timestamp": datetime.now().isoformat()
            }
        )

        print(f"\n✓ Trace logged to LangFuse (duration: {duration:.2f}s)")
        print(f"✓ View trace at: {langfuse.get_trace_url(trace)}")

        return full_response

    except Exception as e:
        # Log the error in LangFuse
        generation.end(
            metadata={
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
        )
        print(f"\n✗ Error: {e}")
        raise


def prompt_titan_with_langfuse(prompt: str, user_id: str = "demo-user"):
    """
    Demonstrate Titan invocation with LangFuse tracing.

    Args:
        prompt: The prompt to send to Titan
        user_id: Identifier for the user making the request
    """
    # Create a trace in LangFuse
    trace = langfuse.trace(
        name="bedrock-titan-call",
        user_id=user_id,
        metadata={
            "model": "amazon.titan-text-lite-v1",
            "provider": "aws-bedrock"
        }
    )

    # Create a generation span within the trace
    generation = trace.generation(
        name="titan-generation",
        model="amazon.titan-text-lite-v1",
        input=prompt,
        metadata={
            "temperature": 0,
            "topP": 1,
            "maxTokenCount": 4096
        }
    )

    start_time = time.time()

    try:
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

        # Collect the full response
        full_response = ""
        for e in response['body']:
            chunk_data = json.loads(e['chunk']['bytes'])
            out_tokens = chunk_data.get('outputText', '')
            full_response += out_tokens
            print(out_tokens, end='')
            sys.stdout.flush()

        print()  # New line after response

        # Calculate duration
        duration = time.time() - start_time

        # Update the generation with the output and metrics
        generation.end(
            output=full_response,
            metadata={
                "latency_seconds": duration,
                "completion_timestamp": datetime.now().isoformat()
            }
        )

        print(f"\n✓ Trace logged to LangFuse (duration: {duration:.2f}s)")
        print(f"✓ View trace at: {langfuse.get_trace_url(trace)}")

        return full_response

    except Exception as e:
        # Log the error in LangFuse
        generation.end(
            metadata={
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
        )
        print(f"\n✗ Error: {e}")
        raise


def prompt_cohere_with_langfuse(prompt: str, user_id: str = "demo-user"):
    """
    Demonstrate Cohere invocation with LangFuse tracing.

    Args:
        prompt: The prompt to send to Cohere
        user_id: Identifier for the user making the request
    """
    # Create a trace in LangFuse
    trace = langfuse.trace(
        name="bedrock-cohere-call",
        user_id=user_id,
        metadata={
            "model": "cohere.command-light-text-v14",
            "provider": "aws-bedrock"
        }
    )

    # Create a generation span within the trace
    generation = trace.generation(
        name="cohere-generation",
        model="cohere.command-light-text-v14",
        input=prompt,
        metadata={
            "stream": True
        }
    )

    start_time = time.time()

    try:
        body = json.dumps({
            "prompt": prompt,
            "stream": True
        })

        response = brt.invoke_model_with_response_stream(
            modelId="cohere.command-light-text-v14",
            body=body
        )

        # Collect the full response
        full_response = ""
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    d = json.loads(chunk.get('bytes').decode())
                    if not d.get("is_finished", False):
                        chunk_text = d.get('text', '')
                        full_response += chunk_text
                        print(chunk_text, end='')
                        sys.stdout.flush()

        print()  # New line after response

        # Calculate duration
        duration = time.time() - start_time

        # Update the generation with the output and metrics
        generation.end(
            output=full_response,
            metadata={
                "latency_seconds": duration,
                "completion_timestamp": datetime.now().isoformat()
            }
        )

        print(f"\n✓ Trace logged to LangFuse (duration: {duration:.2f}s)")
        print(f"✓ View trace at: {langfuse.get_trace_url(trace)}")

        return full_response

    except Exception as e:
        # Log the error in LangFuse
        generation.end(
            metadata={
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
        )
        print(f"\n✗ Error: {e}")
        raise


def demo_comparison():
    """
    Compare multiple models for the same prompt using LangFuse.
    This creates a single trace with multiple generations for comparison.
    """
    prompt = "Human: What are the benefits of using observability tools for LLM applications? Assistant:"

    # Create a trace for the comparison
    trace = langfuse.trace(
        name="model-comparison",
        user_id="demo-user",
        metadata={
            "purpose": "Compare different models",
            "prompt": prompt
        }
    )

    print("=" * 80)
    print("MODEL COMPARISON DEMO")
    print("=" * 80)
    print(f"\nPrompt: {prompt}\n")

    models = [
        ("Claude", "anthropic.claude-instant-v1", "claude-generation"),
        ("Titan", "amazon.titan-text-lite-v1", "titan-generation"),
        ("Cohere", "cohere.command-light-text-v14", "cohere-generation"),
    ]

    for model_name, model_id, gen_name in models:
        print(f"\n--- {model_name} Response ---")

        generation = trace.generation(
            name=gen_name,
            model=model_id,
            input=prompt
        )

        start_time = time.time()

        try:
            # Call the appropriate model based on model_id
            if "claude" in model_id:
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": 500
                })
            elif "titan" in model_id:
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 500,
                        "temperature": 0.7,
                        "topP": 1
                    }
                })
            elif "cohere" in model_id:
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": 500,
                    "stream": True
                })

            response = brt.invoke_model_with_response_stream(
                modelId=model_id,
                body=body
            )

            full_response = ""
            stream = response.get('body')
            if stream:
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        d = json.loads(chunk.get('bytes').decode())

                        # Extract text based on model type
                        if "claude" in model_id:
                            chunk_text = d.get('completion', '')
                        elif "titan" in model_id:
                            chunk_text = d.get('outputText', '')
                        elif "cohere" in model_id:
                            if not d.get("is_finished", False):
                                chunk_text = d.get('text', '')
                            else:
                                chunk_text = ''

                        full_response += chunk_text
                        print(chunk_text, end='')
                        sys.stdout.flush()

            duration = time.time() - start_time

            generation.end(
                output=full_response,
                metadata={
                    "latency_seconds": duration,
                    "model_name": model_name
                }
            )

            print(f"\n(Completed in {duration:.2f}s)")

        except Exception as e:
            duration = time.time() - start_time
            generation.end(
                metadata={
                    "error": str(e),
                    "latency_seconds": duration
                }
            )
            print(f"\n✗ Error with {model_name}: {e}")

    print(f"\n{'=' * 80}")
    print(f"✓ Comparison trace logged to LangFuse")
    print(f"✓ View trace at: {langfuse.get_trace_url(trace)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AWS BEDROCK + LANGFUSE INTEGRATION DEMO")
    print("=" * 80)
    print("\nThis demo shows how to integrate LangFuse observability with AWS Bedrock.\n")

    # Check if LangFuse credentials are configured
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("⚠️  WARNING: LangFuse credentials not found!")
        print("Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file")
        print("See .env.example for details\n")
        sys.exit(1)

    # Example 1: Single Claude call
    print("\n--- Example 1: Claude with LangFuse Tracing ---")
    prompt = "Human: Write a haiku about observability in AI systems. Assistant:"
    prompt_claude_with_langfuse(prompt)

    # Example 2: Titan call
    print("\n\n--- Example 2: Titan with LangFuse Tracing ---")
    prompt_titan = "Write a short explanation of what LangFuse is in one sentence."
    prompt_titan_with_langfuse(prompt_titan)

    # Example 3: Cohere call
    print("\n\n--- Example 3: Cohere with LangFuse Tracing ---")
    prompt_cohere = "List 3 benefits of using AWS Bedrock for LLM applications."
    prompt_cohere_with_langfuse(prompt_cohere)

    # Example 4: Model comparison
    print("\n\n")
    demo_comparison()

    # Flush all pending traces to LangFuse
    langfuse.flush()

    print("\n✓ All traces have been sent to LangFuse")
    print("✓ Visit your LangFuse dashboard to view the traces and analytics\n")
