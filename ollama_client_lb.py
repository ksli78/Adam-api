"""
Ollama Load-Balanced Client - Dual-GPU Support

This wrapper provides load balancing across multiple Ollama instances.
Perfect for dual-GPU setups where each GPU runs independent Ollama server.

Usage:
    # Single host (backward compatible)
    client = OllamaClient(host="http://ollama:11434")

    # Multiple hosts (load balanced)
    client = OllamaClient(hosts=["http://ollama-gpu0:11434", "http://ollama-gpu1:11434"])

    # From environment variable
    import os
    hosts = os.getenv("OLLAMA_HOSTS", "http://localhost:11434").split(",")
    client = OllamaClient(hosts=hosts)

Author: Adam RAG System
Date: 2025
"""

import json
import logging
import random
import requests
from typing import Dict, Any, Iterator, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama client with load balancing support for multiple instances.

    Supports:
    - Single host: OllamaClient(host="http://ollama:11434")
    - Multiple hosts: OllamaClient(hosts=["http://ollama-gpu0:11434", "http://ollama-gpu1:11434"])
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
        Initialize Ollama client with load balancing.

        Args:
            host: Single Ollama server URL (backward compatible)
            hosts: List of Ollama server URLs for load balancing
            timeout: Request timeout in seconds (default: 300 = 5 minutes)
            strategy: Load balancing strategy ("round-robin" or "random")
        """
        # Handle both single host and multiple hosts
        if hosts:
            self.hosts = [h.rstrip('/') for h in hosts]
        elif host:
            self.hosts = [host.rstrip('/')]
        else:
            self.hosts = ["http://localhost:11434"]

        self.timeout = timeout
        self.strategy = strategy
        self.current_index = 0
        self.lock = Lock()  # Thread-safe index for round-robin

        # Track host health (simple circuit breaker)
        self.host_failures = {h: 0 for h in self.hosts}
        self.max_failures = 3  # Mark unhealthy after 3 consecutive failures

        logger.info(
            f"OllamaClient initialized: hosts={self.hosts}, "
            f"strategy={strategy}, timeout={timeout}s"
        )

    def _get_next_host(self) -> str:
        """
        Get next host using configured strategy.

        Returns:
            URL of next Ollama host to use
        """
        # Filter out unhealthy hosts
        healthy_hosts = [
            h for h in self.hosts
            if self.host_failures.get(h, 0) < self.max_failures
        ]

        # If all hosts unhealthy, reset failure counts and try again
        if not healthy_hosts:
            logger.warning("All Ollama hosts marked unhealthy, resetting failure counts")
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
                f"Ollama host {host} marked unhealthy after "
                f"{self.host_failures[host]} failures"
            )

    def _mark_success(self, host: str):
        """Mark a host as successful (reset failure count)."""
        if self.host_failures.get(host, 0) > 0:
            logger.info(f"Ollama host {host} recovered")
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
        Generate text using Ollama with load balancing.

        Automatically retries with different hosts on failure.

        Args:
            model: Model name (e.g., "mistral:7b-instruct-v0.3")
            prompt: Input text prompt
            options: Generation options (temperature, num_predict, etc.)
            stream: Enable streaming response
            keep_alive: Keep model loaded (seconds)

        Returns:
            If stream=False: Response dict with 'response' field
            If stream=True: Generator yielding response chunks
        """
        options = options or {}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }

        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        # Try each host with failover
        for attempt in range(len(self.hosts)):
            host = self._get_next_host()

            try:
                if stream:
                    return self._generate_stream_with_host(host, payload)
                else:
                    return self._generate_complete_with_host(host, payload)

            except Exception as e:
                self._mark_failure(host)
                logger.warning(
                    f"Ollama request failed on {host} (attempt {attempt + 1}/{len(self.hosts)}): {e}"
                )

                # If last attempt, re-raise error
                if attempt == len(self.hosts) - 1:
                    logger.error(f"All Ollama hosts failed after {len(self.hosts)} attempts")
                    raise

        # Should never reach here
        raise RuntimeError("Failed to generate response from any Ollama host")

    def _generate_complete_with_host(self, host: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete response from specific host."""
        generate_url = f"{host}/api/generate"

        try:
            response = requests.post(
                generate_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            data = response.json()

            if "response" in data:
                self._mark_success(host)
                logger.debug(f"Generated {len(data['response'])} chars from {host}")
                return data
            else:
                logger.error(f"Unexpected Ollama response format from {host}: {data}")
                self._mark_failure(host)
                raise ValueError(f"Unexpected response format from {host}")

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out on {host} after {self.timeout}s")
            self._mark_failure(host)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed on {host}: {e}")
            self._mark_failure(host)
            raise

    def _generate_stream_with_host(self, host: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Generate streaming response from specific host."""
        generate_url = f"{host}/api/generate"

        try:
            with requests.post(
                generate_url,
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

                    try:
                        data = json.loads(line)
                        yield data

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse streaming chunk from {host}: {e}")
                        continue

        except requests.exceptions.Timeout:
            logger.error(f"Ollama streaming timed out on {host} after {self.timeout}s")
            self._mark_failure(host)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama streaming failed on {host}: {e}")
            self._mark_failure(host)
            raise

    def health_check(self, host: Optional[str] = None) -> bool:
        """
        Check health of Ollama host(s).

        Args:
            host: Specific host to check, or None to check all

        Returns:
            True if host(s) healthy, False otherwise
        """
        hosts_to_check = [host] if host else self.hosts

        for h in hosts_to_check:
            try:
                response = requests.get(f"{h}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info(f"Ollama host {h} is healthy")
                    self._mark_success(h)
                else:
                    logger.warning(f"Ollama host {h} returned status {response.status_code}")
                    self._mark_failure(h)
                    return False
            except Exception as e:
                logger.error(f"Ollama host {h} health check failed: {e}")
                self._mark_failure(h)
                return False

        return True

    def get_host_status(self) -> Dict[str, Any]:
        """
        Get status of all Ollama hosts.

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
Client = OllamaClient


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Single host
    client1 = OllamaClient(host="http://localhost:11434")

    # Multiple hosts (load balanced)
    client2 = OllamaClient(
        hosts=["http://localhost:11434", "http://localhost:11435"],
        strategy="round-robin"
    )

    # Test health
    print("Host status:", client2.get_host_status())

    # Test generation (non-streaming)
    response = client2.generate(
        model="mistral:7b-instruct-v0.3",
        prompt="What is 2+2?",
        options={"temperature": 0.7, "num_predict": 50}
    )
    print(f"Response: {response['response']}")
