# AWS Bedrock LLM Examples with LangFuse Integration

This repository demonstrates how to use AWS Bedrock with various LLM providers (Claude, Titan, Cohere, Llama) and integrate LangFuse for observability and tracing.

## Features

- **AWS Bedrock Integration**: Examples for multiple LLM providers
  - Anthropic Claude
  - Amazon Titan
  - Cohere Command
  - Meta Llama 2

- **LangFuse Observability**: Track and monitor your LLM applications
  - Trace LLM calls end-to-end
  - Monitor latency and performance
  - Track costs and token usage
  - Debug and optimize prompts
  - Analyze user interactions

## What is LangFuse?

[LangFuse](https://langfuse.com/) is an open-source observability and analytics platform for LLM applications. It helps you:

- 📊 **Monitor**: Track all LLM calls, latency, and costs in real-time
- 🔍 **Debug**: Inspect traces to understand what's happening in your LLM pipeline
- 📈 **Analyze**: Get insights into usage patterns and model performance
- 🎯 **Optimize**: Compare different models and prompts to improve results
- 🔄 **Version**: Manage and track different versions of your prompts

## Prerequisites

- Python 3.7+
- AWS Account with Bedrock access
- AWS credentials configured (via AWS CLI or environment variables)
- LangFuse account (free tier available at [cloud.langfuse.com](https://cloud.langfuse.com))

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hello-bedrock
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**:

   Make sure you have AWS credentials configured. You can use AWS CLI:
   ```bash
   aws configure
   ```

   Or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Set up LangFuse**:

   a. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com)

   b. Navigate to **Settings → API Keys** and create a new API key pair

   c. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

   d. Edit `.env` and add your LangFuse credentials:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

## Usage

### Basic Examples (without LangFuse)

Run the basic examples to test AWS Bedrock connectivity:

```bash
python main.py
```

This will prompt Claude to generate a 7-day itinerary for Japan. You can uncomment other model calls in `main.py` to test different providers.

### LangFuse Integration Demo

Run the LangFuse demo to see observability in action:

```bash
python langfuse_demo.py
```

This demo includes:

1. **Single model calls with tracing**: See how individual LLM calls are tracked
2. **Model comparison**: Compare responses from Claude, Titan, and Cohere for the same prompt
3. **Performance metrics**: View latency and other metrics in your LangFuse dashboard

After running the demo, visit your [LangFuse dashboard](https://cloud.langfuse.com) to:
- View detailed traces of each LLM call
- Analyze performance metrics
- Compare different models
- Debug any issues

## Project Structure

```
hello-bedrock/
├── main.py                 # Basic AWS Bedrock examples
├── langfuse_demo.py        # LangFuse integration demo
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .env                   # Your credentials (not committed)
└── README.md             # This file
```

## Code Examples

### Basic Bedrock Call (from main.py)

```python
import boto3
import json

brt = boto3.client(service_name='bedrock-runtime')

body = json.dumps({
    "prompt": "Human: Hello! Assistant:",
    "max_tokens_to_sample": 4000
})

response = brt.invoke_model_with_response_stream(
    modelId="anthropic.claude-instant-v1",
    body=body
)
```

### Bedrock + LangFuse (from langfuse_demo.py)

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create a trace
trace = langfuse.trace(
    name="bedrock-claude-call",
    user_id="user-123"
)

# Create a generation within the trace
generation = trace.generation(
    name="claude-generation",
    model="anthropic.claude-instant-v1",
    input=prompt
)

# Make your LLM call
response = brt.invoke_model_with_response_stream(...)

# Update with results
generation.end(
    output=full_response,
    metadata={"latency_seconds": duration}
)
```

## LangFuse Features in Action

### 1. Trace Visualization
Each LLM call creates a trace in LangFuse showing:
- Input prompt
- Model used
- Output generated
- Latency
- Timestamp

### 2. Model Comparison
The demo includes a comparison function that:
- Sends the same prompt to multiple models
- Tracks all responses in a single trace
- Allows side-by-side comparison in the dashboard

### 3. Performance Monitoring
Track metrics like:
- Average response time per model
- Token usage and costs
- Error rates
- User patterns

### 4. Debug and Optimize
Use LangFuse to:
- Identify slow or failing requests
- Compare different prompt versions
- A/B test model parameters
- Analyze user feedback

## Self-Hosting LangFuse (Optional)

If you prefer to run LangFuse locally or on your own infrastructure:

1. Follow the [LangFuse self-hosting guide](https://langfuse.com/docs/deployment/self-host)

2. Update your `.env` file with your local instance URL:
   ```
   LANGFUSE_HOST=http://localhost:3000
   ```

## AWS Bedrock Model Access

Make sure you have enabled model access in AWS Bedrock:

1. Go to AWS Console → Bedrock
2. Navigate to "Model access" in the left sidebar
3. Request access for the models you want to use:
   - Anthropic Claude
   - Amazon Titan
   - Cohere Command
   - Meta Llama 2

Model access requests are usually approved within a few minutes.

## Troubleshooting

### LangFuse Connection Issues

If you see "LangFuse credentials not found":
- Make sure you've copied `.env.example` to `.env`
- Verify your API keys are correct
- Check that there are no extra spaces in your `.env` file

### AWS Bedrock Errors

**"AccessDeniedException"**:
- Ensure you have requested model access in AWS Bedrock console
- Verify your AWS credentials have the necessary permissions

**"ModelNotAvailable"**:
- Some models are only available in specific regions
- Try `us-east-1` or `us-west-2`

**"ThrottlingException"**:
- You're hitting rate limits
- Add delays between requests or request a quota increase

## Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LangFuse Documentation](https://langfuse.com/docs)
- [LangFuse GitHub](https://github.com/langfuse/langfuse)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

## Contributing

Feel free to submit issues or pull requests to improve these examples!

## License

This project is provided as-is for educational and demonstration purposes.
