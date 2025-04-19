"""
OpenAI adapter for ACDEK v3.

This adapter provides integration with OpenAI API for generating
text responses that can be evaluated by the grounding engine.
"""
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union

import openai
from openai import OpenAI

from registry.plugin_registry import register_plugin
from registry.registry_types import PluginType
from interfaces.llm_adapter_interface import LLMAdapterInterface

logger = logging.getLogger(__name__)

@register_plugin(
    plugin_type=PluginType.LLM_ADAPTER,
    name="openai_adapter",
    version="1.0.0",
    description="OpenAI API adapter for generating responses to be evaluated by the grounding engine",
    interfaces=["LLMAdapterInterface"],
    is_default=True
)
class OpenAIAdapter(LLMAdapterInterface):
    """
    OpenAI API adapter for generating text responses.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize the OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (defaults to gpt-4)
            temperature: Sampling temperature (0-1, higher is more creative)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key.")
            
        self.model = model
        self.temperature = temperature
        self.client = None
        
        # Metrics tracking
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        
        # Initialize client if API key is available
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model {model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                self.error_count += 1
    
    def call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API to generate a response.
        
        Args:
            prompt: Text prompt to send to the API
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If no API key is available or API call fails
        """
        if not self.api_key or not self.client:
            raise RuntimeError("OpenAI API key not available")
        
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            # Update metrics
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            # Log usage
            logger.info(
                f"OpenAI API call - model: {self.model}, tokens: {response.usage.total_tokens}, "
                f"latency: {latency:.2f}s"
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def call_with_system_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call OpenAI API with separate system and user prompts.
        
        Args:
            system_prompt: System message for framing the interaction
            user_prompt: User message/question
            
        Returns:
            Generated text response
        """
        if not self.api_key or not self.client:
            raise RuntimeError("OpenAI API key not available")
        
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            # Update metrics
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def call_with_accuracy_focus(self, prompt: str) -> str:
        """
        Call OpenAI API with a focus on accuracy (low temperature).
        
        Args:
            prompt: Text prompt to send to the API
            
        Returns:
            Generated text response aimed at high accuracy
        """
        if not self.api_key or not self.client:
            raise RuntimeError("OpenAI API key not available")
        
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate assistant focused on providing factual information. Always prioritize accuracy and indicate when you're uncertain."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for higher accuracy
                max_tokens=1000
            )
            
            # Update metrics
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def call_with_creativity_focus(self, prompt: str) -> str:
        """
        Call OpenAI API with a focus on creativity (high temperature).
        
        Args:
            prompt: Text prompt to send to the API
            
        Returns:
            Generated text response with more creativity
        """
        if not self.api_key or not self.client:
            raise RuntimeError("OpenAI API key not available")
        
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative assistant that provides imaginative and engaging responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # High temperature for more creativity
                max_tokens=1000
            )
            
            # Update metrics
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def call_with_hallucination_risk(self, prompt: str) -> str:
        """
        Call OpenAI API with parameters that increase hallucination risk.
        
        This method is specifically for testing the grounding engine's
        ability to detect hallucinations.
        
        Args:
            prompt: Text prompt to send to the API
            
        Returns:
            Generated text response with higher hallucination risk
        """
        if not self.api_key or not self.client:
            raise RuntimeError("OpenAI API key not available")
        
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative assistant that doesn't need to verify information. Feel free to make up details to make your answer more interesting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,  # Maximum temperature
                presence_penalty=0.6,  # Encourage novelty
                frequency_penalty=0.1,  # Don't discourage repetition too much
                max_tokens=1000
            )
            
            # Update metrics
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics for the adapter.
        
        Returns:
            Dictionary with usage metrics
        """
        avg_latency = self.total_latency / max(1, self.request_count)
        avg_tokens = self.total_tokens / max(1, self.request_count)
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "total_latency": self.total_latency,
            "avg_latency_seconds": avg_latency,
            "avg_tokens_per_request": avg_tokens,
            "model": self.model
        }