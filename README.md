# Mixture-of-Models Gateway

## Overview
Mixture-of-Models (MoM) is an advanced pattern that queries multiple large language models simultaneously and synthesizes their responses into a single high-quality output. By leveraging diverse model capabilities and reasoning paths, this approach discovers novel solutions and consistently produces superior results compared to individual models. The technique has demonstrated particular effectiveness in coding tasks, where combined outputs yield improved accuracy, completeness, and creativity by resolving contradictions and filling knowledge gaps across responses.

## Key Benefits
- Parallel execution uncovers varied approaches and perspectives across different models
- Synthesized responses combine strengths of specialized models while mitigating individual weaknesses
- Collective intelligence identifies errors and contradictions while preserving valuable insights
- Cross-pollination of concepts from different model architectures reveals innovative solutions

## Quick Start Guide
1. Install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure models in `config.yaml` with your API keys (you can define environment variables for API keys):
   ```yaml
   models:
     - name: "openrouter"
       api_key: "${OPENROUTER_API_KEY}"
   ```

3. Start the gateway service:
   ```bash
   python main.py --debug-requests
   ```

4. Connect [aider](https://aider.chat) using:
   ```bash
   LM_STUDIO_API_BASE=http://127.0.0.1:8000/v1 \
   LM_STUDIO_API_KEY=123 \
   aider --model lm_studio/mom --no-stream --editor-model gpt-4.1 --architect --weak-model gpt-4.1-mini
   ```

**Security Warning**: This gateway performs no API key validation. Never expose it beyond localhost as it lacks authentication and security measures suitable for production.

## Supported Features
- OpenAI-compatible `/v1/chat/completions` endpoint
- Parallel fan-out to multiple models/configurable via YAML
- Automatic response synthesis through specialized critic model
- Environment variable substitution in configuration
- Debug request tracing (saved to `debug-requests/`)
- Customizable model parameters (temperature, max_tokens)
- Automatic retries with exponential backoff
- Timeout handling for upstream API calls

## Current Limitations
- Streaming responses (`"stream": true`) are unsupported and return HTTP 400 errors
- Limited to chat completion endpoints (no embeddings, images, or other modalities)
- No authentication, rate limiting, or production hardening
- Static configuration requiring service restart for changes
- Basic error handling with no advanced fallback mechanisms
- Critic uses single-stage prompting without multi-step verification
- Proof-of-concept implementation not suitable for production workloads

## Contributing
This proof-of-concept project encourages exploration of MoM techniques. Any suggestions, bug reports, and pull requests are welcome.

Share findings from experiments with model combinations, critic approaches, or evaluation metrics to advance MoM research.
