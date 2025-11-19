"""
Text Generation Inference (TGI) Client - Drop-in Replacement for Ollama

This client mimics the Ollama client interface but connects to HuggingFace's
Text Generation Inference (TGI) server instead.

TGI has better support for Turing architecture GPUs (Quadro RTX 5000) than vLLM.

Usage:
    # Replace Ollama client
    from tgi_client import TGIClient as Client

    # Same interface as before
    client = Client(host="http://tgi-server:80")
    response = client.generate(
        model="ibm-granite/granite-3.1-8b-instruct",  # Ignored by TGI
        prompt="What is the PTO policy?",
        options={"temperature": 0.7}
    )
    print(response.response)

Author: Adam RAG System
Date: 2025
"""

import json
import logging
import requests
from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional

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


class TGIClient:
    """
    Text Generation Inference (TGI) client mimicking Ollama interface.

    TGI is HuggingFace's production inference server with excellent
    support for Turing architecture GPUs.

    Better alternative to vLLM for Quadro RTX 5000.
    """

    def __init__(
        self,
        host: str = "http://localhost:80",
        timeout: int = 300
    ):
        """
        Initialize TGI client.

        Args:
            host: TGI server URL (default: http://localhost:80)
            timeout: Request timeout in seconds (default: 300 = 5 minutes)
        """
        self.host = host.rstrip('/')
        self.timeout = timeout

        logger.info(f"TGIClient initialized: host={host}, timeout={timeout}s")

    def generate(
        self,
        model: str,  # Ignored - TGI loads one model at startup
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[int] = None  # Ignored - TGI keeps model loaded
    ) -> Any:
        """
        Generate text using TGI.

        Args:
            model: Model name (IGNORED - TGI loads model at startup)
            prompt: Input text prompt
            options: Generation options (temperature, max_tokens, etc.)
            stream: Enable streaming response
            keep_alive: Ignored (TGI keeps models loaded)

        Returns:
            If stream=False: GenerateResponse object
            If stream=True: Generator yielding GenerateResponse objects
        """
        options = options or {}

        # Build TGI parameters
        parameters = {
            "temperature": options.get("temperature", 0.7),
            "max_new_tokens": options.get("num_predict", 2000),
            "top_p": options.get("top_p", 1.0),
            "do_sample": options.get("temperature", 0.7) > 0,  # Deterministic if temp=0
        }

        # Add optional parameters if provided
        if "top_k" in options:
            parameters["top_k"] = options["top_k"]
        if "repetition_penalty" in options:
            parameters["repetition_penalty"] = options["repetition_penalty"]
        if "stop" in options:
            parameters["stop_sequences"] = options["stop"]

        payload = {
            "inputs": prompt,
            "parameters": parameters
        }

        if stream:
            payload["stream"] = True
            return self._generate_stream(payload)
        else:
            return self._generate_complete(payload)

    def _generate_complete(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate complete response from TGI."""
        try:
            response = requests.post(
                f"{self.host}/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()

            # TGI returns: {"generated_text": "..."}
            text = data.get("generated_text", "")

            logger.debug(f"Generated {len(text)} chars from TGI")
            return GenerateResponse(response=text, done=True)

        except requests.exceptions.Timeout:
            logger.error(f"TGI request timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"TGI request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in TGI generation: {e}")
            raise

    def _generate_stream(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response from TGI."""
        try:
            with requests.post(
                f"{self.host}/generate_stream",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
                stream=True
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode('utf-8')

                    # TGI streams: data: {...}
                    if line_str.startswith('data:'):
                        data_str = line_str[5:].strip()

                        try:
                            data = json.loads(data_str)

                            # TGI stream format:
                            # {"token": {"text": "...", "special": false}, "generated_text": null}
                            # Last message: {"token": {...}, "generated_text": "full text"}

                            token = data.get("token", {}).get("text", "")
                            generated_text = data.get("generated_text")

                            if token:
                                # Yield token
                                yield GenerateResponse(
                                    response=token,
                                    done=(generated_text is not None)
                                )

                            if generated_text is not None:
                                # Final message - done
                                yield GenerateResponse(response="", done=True)
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming chunk: {e}")
                            continue

        except requests.exceptions.Timeout:
            logger.error(f"TGI streaming timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"TGI streaming failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in TGI streaming: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check health of TGI server.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"TGI server {self.host} is healthy")
                return True
            else:
                logger.warning(f"TGI server {self.host} returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"TGI server {self.host} health check failed: {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get TGI server information.

        Returns:
            Dict with server info (model, version, etc.)
        """
        try:
            response = requests.get(f"{self.host}/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get TGI server info: {e}")
            return {}


# Backward compatibility alias
Client = TGIClient


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize client
    client = TGIClient(host="http://localhost:8001")

    # Health check
    if client.health_check():
        print("✅ TGI server is healthy")

        # Get server info
        info = client.get_server_info()
        print(f"Server info: {info}")

        # Test generation (non-streaming)
        print("\nTesting non-streaming generation...")
        response = client.generate(
            model="ibm-granite/granite-3.1-8b-instruct",  # Ignored by TGI
            prompt="What is 2+2? Answer briefly.",
            options={"temperature": 0.7, "num_predict": 50}
        )
        print(f"Response: {response.response}")

        # Test generation (streaming)
        print("\nTesting streaming generation...")
        stream = client.generate(
            model="ibm-granite/granite-3.1-8b-instruct",
            prompt="Count from 1 to 5.",
            options={"temperature": 0.7, "num_predict": 50},
            stream=True
        )
        print("Streaming response: ", end="")
        for chunk in stream:
            print(chunk.response, end="", flush=True)
        print()
    else:
        print("❌ TGI server is not available")
