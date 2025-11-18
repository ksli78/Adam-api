"""
vLLM Client Wrapper - Drop-in replacement for Ollama Client

This wrapper provides an Ollama-compatible interface for vLLM's OpenAI-compatible API.
Allows migration from Ollama to vLLM with minimal code changes.

Author: Adam RAG System
Date: 2025
"""

import json
import logging
from typing import Dict, Any, Iterator, Optional
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerateResponse:
    """
    Response object mimicking Ollama's GenerateResponse.

    Attributes:
        response: The generated text token(s)
        done: Whether generation is complete
    """
    response: str
    done: bool = False


class VLLMClient:
    """
    vLLM client that mimics Ollama's client interface.

    This allows drop-in replacement:
    - Before: ollama.Client(host="http://ollama:11434")
    - After:  VLLMClient(host="http://vllm:8000")

    Supports both streaming and non-streaming generation.
    """

    def __init__(self, host: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize vLLM client.

        Args:
            host: vLLM server URL (OpenAI-compatible endpoint)
            timeout: Request timeout in seconds (default: 300 = 5 minutes)
        """
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.completions_url = f"{self.host}/v1/completions"

        logger.info(f"VLLMClient initialized: host={self.host}, timeout={timeout}s")

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[int] = None  # Ignored - vLLM keeps models loaded
    ) -> Any:
        """
        Generate text using vLLM (mimics ollama.Client.generate).

        Args:
            model: Model name (e.g., "mistralai/Mistral-Small-Instruct-2409")
            prompt: Input text prompt
            options: Generation options (temperature, max_tokens, etc.)
            stream: Enable streaming response
            keep_alive: Ignored (vLLM keeps models loaded automatically)

        Returns:
            If stream=False: GenerateResponse object with full text
            If stream=True: Generator yielding GenerateResponse objects
        """
        # Convert Ollama options to OpenAI-compatible parameters
        options = options or {}

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options.get("temperature", 0.7),
            "max_tokens": options.get("num_predict", 2000),
            "stream": stream,
            "top_p": options.get("top_p", 1.0),
            "n": 1,  # Number of completions
        }

        # Add optional parameters if provided
        if "num_ctx" in options:
            # num_ctx in Ollama = context window, not directly mapped to OpenAI API
            # vLLM uses max_model_len at server startup
            pass  # Handled by --max-model-len in vLLM server config

        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_complete(payload)

    def _generate_complete(self, payload: Dict[str, Any]) -> GenerateResponse:
        """
        Generate complete response (non-streaming).

        Args:
            payload: Request payload for vLLM API

        Returns:
            GenerateResponse with full generated text
        """
        try:
            response = requests.post(
                self.completions_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()

            # Extract text from OpenAI-compatible response
            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0]["text"]
                return GenerateResponse(response=text, done=True)
            else:
                logger.error(f"Unexpected vLLM response format: {data}")
                return GenerateResponse(response="", done=True)

        except requests.exceptions.Timeout:
            logger.error(f"vLLM request timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in vLLM generate: {e}")
            raise

    def _generate_stream(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """
        Generate streaming response (yields tokens as they're generated).

        Args:
            payload: Request payload for vLLM API

        Yields:
            GenerateResponse objects with individual tokens
        """
        try:
            with requests.post(
                self.completions_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
                stream=True
            ) as response:
                response.raise_for_status()

                # Process Server-Sent Events (SSE) stream
                for line in response.iter_lines():
                    if not line:
                        continue

                    # Decode and remove "data: " prefix
                    line_str = line.decode('utf-8')

                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove "data: " prefix

                        # Check for stream end marker
                        if data_str.strip() == '[DONE]':
                            yield GenerateResponse(response="", done=True)
                            break

                        try:
                            data = json.loads(data_str)

                            # Extract token from OpenAI-compatible streaming response
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                token = choice.get("text", "")
                                finish_reason = choice.get("finish_reason")

                                if token:
                                    yield GenerateResponse(
                                        response=token,
                                        done=(finish_reason is not None)
                                    )

                                # If stream is finished, yield final done response
                                if finish_reason:
                                    yield GenerateResponse(response="", done=True)
                                    break

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming chunk: {data_str[:100]}")
                            continue

        except requests.exceptions.Timeout:
            logger.error(f"vLLM streaming request timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM streaming request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in vLLM streaming: {e}")
            raise

    def chat(
        self,
        model: str,
        messages: list,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ):
        """
        Chat completion (not implemented - RAG system uses generate()).

        Note: This can be implemented if needed by calling /v1/chat/completions
        """
        raise NotImplementedError(
            "VLLMClient.chat() not implemented. "
            "Use generate() instead, or implement chat endpoint."
        )


# Backward compatibility alias
Client = VLLMClient


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Non-streaming generation
    client = VLLMClient(host="http://localhost:8001")

    print("Testing non-streaming generation...")
    response = client.generate(
        model="mistralai/Mistral-Small-Instruct-2409",
        prompt="What is the capital of France?",
        options={"temperature": 0.7, "num_predict": 100}
    )
    print(f"Response: {response.response}")

    print("\nTesting streaming generation...")
    for chunk in client.generate(
        model="mistralai/Mistral-Small-Instruct-2409",
        prompt="Count from 1 to 5:",
        options={"temperature": 0.7, "num_predict": 50},
        stream=True
    ):
        if chunk.response:
            print(chunk.response, end='', flush=True)
    print("\n\nDone!")
