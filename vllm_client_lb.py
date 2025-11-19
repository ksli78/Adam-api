"""
vLLM Load-Balanced Client - Multi-GPU Support

This wrapper provides load balancing across multiple vLLM instances.
Perfect for dual-GPU setups where each GPU runs independent vLLM server.

Usage:
    # Single host (backward compatible)
    client = VLLMClient(host="http://vllm:8000")

    # Multiple hosts (load balanced)
    client = VLLMClient(hosts=["http://vllm-gpu0:8000", "http://vllm-gpu1:8000"])

    # From environment variable
    import os
    hosts = os.getenv("VLLM_HOSTS", "http://localhost:8000").split(",")
    client = VLLMClient(hosts=hosts)

Author: Adam RAG System
Date: 2025
"""

import json
import logging
import random
import time
from typing import Dict, Any, Iterator, Optional, List
import requests
from dataclasses import dataclass
from threading import Lock

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
    vLLM client with load balancing support for multiple instances.

    Supports:
    - Single host: VLLMClient(host="http://vllm:8000")
    - Multiple hosts: VLLMClient(hosts=["http://vllm-gpu0:8000", "http://vllm-gpu1:8000"])
    - Round-robin load balancing
    - Automatic failover to healthy instances
    """

    def __init__(
        self,
        host: Optional[str] = None,
        hosts: Optional[List[str]] = None,
        timeout: int = 300,
        strategy: str = "round-robin"
    ):
        """
        Initialize vLLM client with load balancing.

        Args:
            host: Single vLLM server URL (backward compatible)
            hosts: List of vLLM server URLs for load balancing
            timeout: Request timeout in seconds (default: 300 = 5 minutes)
            strategy: Load balancing strategy ("round-robin" or "random")
        """
        # Handle both single host and multiple hosts
        if hosts:
            self.hosts = [h.rstrip('/') for h in hosts]
        elif host:
            self.hosts = [host.rstrip('/')]
        else:
            self.hosts = ["http://localhost:8000"]

        self.timeout = timeout
        self.strategy = strategy
        self.current_index = 0
        self.lock = Lock()  # Thread-safe index for round-robin

        # Track host health (simple circuit breaker)
        self.host_failures = {h: 0 for h in self.hosts}
        self.max_failures = 3  # Mark unhealthy after 3 consecutive failures

        logger.info(
            f"VLLMClient initialized: hosts={self.hosts}, "
            f"strategy={strategy}, timeout={timeout}s"
        )

    def _get_next_host(self) -> str:
        """
        Get next host using configured strategy.

        Returns:
            URL of next vLLM host to use
        """
        # Filter out unhealthy hosts
        healthy_hosts = [
            h for h in self.hosts
            if self.host_failures.get(h, 0) < self.max_failures
        ]

        # If all hosts unhealthy, reset failure counts and try again
        if not healthy_hosts:
            logger.warning("All vLLM hosts marked unhealthy, resetting failure counts")
            self.host_failures = {h: 0 for h in self.hosts}
            healthy_hosts = self.hosts

        if self.strategy == "random":
            return random.choice(healthy_hosts)
        else:  # round-robin
            with self.lock:
                # Find next healthy host
                for _ in range(len(self.hosts)):
                    host = self.hosts[self.current_index % len(self.hosts)]
                    self.current_index += 1
                    if host in healthy_hosts:
                        return host
                # Fallback to first healthy host
                return healthy_hosts[0]

    def _mark_failure(self, host: str):
        """Mark a host as failed (for circuit breaker)."""
        self.host_failures[host] = self.host_failures.get(host, 0) + 1
        if self.host_failures[host] >= self.max_failures:
            logger.warning(
                f"vLLM host {host} marked unhealthy after "
                f"{self.host_failures[host]} failures"
            )

    def _mark_success(self, host: str):
        """Mark a host as successful (reset failure count)."""
        if self.host_failures.get(host, 0) > 0:
            logger.info(f"vLLM host {host} recovered")
        self.host_failures[host] = 0

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[int] = None
    ) -> Any:
        """
        Generate text using vLLM with load balancing.

        Automatically retries with different hosts on failure.

        Args:
            model: Model name
            prompt: Input text prompt
            options: Generation options (temperature, max_tokens, etc.)
            stream: Enable streaming response
            keep_alive: Ignored (vLLM keeps models loaded)

        Returns:
            If stream=False: GenerateResponse object
            If stream=True: Generator yielding GenerateResponse objects
        """
        options = options or {}

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": options.get("temperature", 0.7),
            "max_tokens": options.get("num_predict", 2000),
            "stream": stream,
            "top_p": options.get("top_p", 1.0),
            "n": 1,
        }

        # Try each host with failover
        for attempt in range(len(self.hosts)):
            host = self._get_next_host()
            completions_url = f"{host}/v1/completions"

            try:
                if stream:
                    return self._generate_stream_with_host(host, payload)
                else:
                    return self._generate_complete_with_host(host, payload)

            except Exception as e:
                self._mark_failure(host)
                logger.warning(
                    f"vLLM request failed on {host} (attempt {attempt + 1}/{len(self.hosts)}): {e}"
                )

                # If last attempt, re-raise error
                if attempt == len(self.hosts) - 1:
                    logger.error(f"All vLLM hosts failed after {len(self.hosts)} attempts")
                    raise

        # Should never reach here
        raise RuntimeError("Failed to generate response from any vLLM host")

    def _generate_complete_with_host(
        self,
        host: str,
        payload: Dict[str, Any]
    ) -> GenerateResponse:
        """Generate complete response from specific host."""
        completions_url = f"{host}/v1/completions"

        try:
            response = requests.post(
                completions_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0]["text"]
                self._mark_success(host)
                logger.debug(f"Generated {len(text)} chars from {host}")
                return GenerateResponse(response=text, done=True)
            else:
                logger.error(f"Unexpected vLLM response format from {host}: {data}")
                self._mark_failure(host)
                raise ValueError(f"Unexpected response format from {host}")

        except requests.exceptions.Timeout:
            logger.error(f"vLLM request timed out on {host} after {self.timeout}s")
            self._mark_failure(host)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed on {host}: {e}")
            self._mark_failure(host)
            raise

    def _generate_stream_with_host(
        self,
        host: str,
        payload: Dict[str, Any]
    ) -> Iterator[GenerateResponse]:
        """Generate streaming response from specific host."""
        completions_url = f"{host}/v1/completions"

        try:
            with requests.post(
                completions_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
                stream=True
            ) as response:
                response.raise_for_status()
                self._mark_success(host)

                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode('utf-8')

                    if line_str.startswith('data: '):
                        data_str = line_str[6:]

                        if data_str.strip() == '[DONE]':
                            yield GenerateResponse(response="", done=True)
                            break

                        try:
                            data = json.loads(data_str)

                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                token = choice.get("text", "")
                                finish_reason = choice.get("finish_reason")

                                if token:
                                    yield GenerateResponse(
                                        response=token,
                                        done=(finish_reason is not None)
                                    )

                                if finish_reason:
                                    yield GenerateResponse(response="", done=True)
                                    break

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming chunk from {host}")
                            continue

        except requests.exceptions.Timeout:
            logger.error(f"vLLM streaming timed out on {host} after {self.timeout}s")
            self._mark_failure(host)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM streaming failed on {host}: {e}")
            self._mark_failure(host)
            raise

    def health_check(self, host: Optional[str] = None) -> bool:
        """
        Check health of vLLM host(s).

        Args:
            host: Specific host to check, or None to check all

        Returns:
            True if host(s) healthy, False otherwise
        """
        hosts_to_check = [host] if host else self.hosts

        for h in hosts_to_check:
            try:
                response = requests.get(f"{h}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM host {h} is healthy")
                    self._mark_success(h)
                else:
                    logger.warning(f"vLLM host {h} returned status {response.status_code}")
                    self._mark_failure(h)
                    return False
            except Exception as e:
                logger.error(f"vLLM host {h} health check failed: {e}")
                self._mark_failure(h)
                return False

        return True

    def get_host_status(self) -> Dict[str, Any]:
        """
        Get status of all vLLM hosts.

        Returns:
            Dict with host health information
        """
        return {
            "hosts": self.hosts,
            "failures": self.host_failures,
            "healthy_hosts": [
                h for h in self.hosts
                if self.host_failures.get(h, 0) < self.max_failures
            ],
            "strategy": self.strategy
        }


# Backward compatibility alias
Client = VLLMClient


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Single host
    client1 = VLLMClient(host="http://localhost:8001")

    # Multiple hosts (load balanced)
    client2 = VLLMClient(
        hosts=["http://localhost:8001", "http://localhost:8002"],
        strategy="round-robin"
    )

    # Test health
    print("Host status:", client2.get_host_status())

    # Test generation
    response = client2.generate(
        model="ibm-granite/granite-3.1-8b-instruct",
        prompt="What is 2+2?",
        options={"temperature": 0.7, "num_predict": 50}
    )
    print(f"Response: {response.response}")
