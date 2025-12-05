"""
LLM Oracle Diagnostic Engine
Implements the core logic for ontology alignment using LLM-based oracle
"""

import os
import json
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from .models import EntityMapping, DiagnosticResult
from .prompt_builder import PNLFSPromptBuilder, SYSTEM_PROMPTS


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """
        Send diagnostic prompt to LLM and get True/False response

        Args:
            prompt: The PNLF^S formatted prompt
            system_prompt: Optional system prompt

        Returns:
            Boolean diagnosis result
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (GPT-4o, GPT-4o Mini, etc.)"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize OpenAI provider

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """Send prompt to OpenAI and get structured boolean response"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use structured output to ensure boolean response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnostic_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "boolean",
                                "description": "True if entities mean the same thing, False otherwise"
                            }
                        },
                        "required": ["result"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.0
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        return result_json["result"]


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic provider

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """Send prompt to Claude and parse boolean response"""

        # Enhance prompt to ensure True/False response
        enhanced_prompt = f"{prompt}\n\nRespond with ONLY the word 'True' or 'False', nothing else."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=0.0,
            system=system_prompt if system_prompt else "You are a helpful assistant.",
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ]
        )

        result_text = response.content[0].text.strip()

        # Parse response
        if "true" in result_text.lower():
            return True
        elif "false" in result_text.lower():
            return False
        else:
            raise ValueError(f"Unexpected response from LLM: {result_text}")


class GeminiProvider(LLMProvider):
    """Google Gemini provider (Recommended: Gemini Flash 2.5)"""

    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize Gemini provider

        Args:
            model: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-flash")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not provided")

        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "boolean"
                            }
                        },
                        "required": ["result"]
                    }
                }
            )
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """Send prompt to Gemini and get structured boolean response"""

        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.client.generate_content(full_prompt)
        result_json = json.loads(response.text)
        return result_json["result"]


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls"""

    def __init__(self, default_response: bool = True):
        """
        Initialize mock provider

        Args:
            default_response: Default boolean response to return
        """
        self.default_response = default_response

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """Return mock response"""
        print(f"[MOCK] Prompt: {prompt[:100]}...")
        return self.default_response


class OntologyAlignmentOracle:
    """
    Main LLM Oracle for Ontology Alignment

    Implements the diagnostic engine as specified in instructions.md
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt_type: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ):
        """
        Initialize the oracle

        Args:
            provider: LLM provider instance
            system_prompt_type: Type of system prompt to use
                               ("biomedical_specialist", "explicit_semantics", "combined", or None)
            custom_system_prompt: Custom system prompt (overrides system_prompt_type)
        """
        self.provider = provider
        self.prompt_builder = PNLFSPromptBuilder()

        # Set system prompt
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        elif system_prompt_type:
            self.system_prompt = SYSTEM_PROMPTS.get(system_prompt_type)
        else:
            self.system_prompt = None

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        """
        Diagnose a single entity mapping

        Args:
            mapping: EntityMapping to diagnose

        Returns:
            DiagnosticResult with prediction
        """
        # Build PNLF^S prompt
        prompt = self.prompt_builder.build_prompt(mapping)

        # Get LLM diagnosis
        prediction = self.provider.diagnose(prompt, self.system_prompt)

        # Create result
        result = DiagnosticResult(
            mapping=mapping,
            prediction=prediction,
            raw_response=str(prediction)
        )

        return result

    def diagnose_batch(self, mappings: list[EntityMapping]) -> list[DiagnosticResult]:
        """
        Diagnose multiple mappings

        Args:
            mappings: List of EntityMappings to diagnose

        Returns:
            List of DiagnosticResults
        """
        results = []
        for mapping in mappings:
            result = self.diagnose_mapping(mapping)
            results.append(result)
        return results

    def evaluate_performance(self, results: list[DiagnosticResult]) -> Dict[str, float]:
        """
        Evaluate diagnostic performance using standard metrics

        Args:
            results: List of DiagnosticResults with ground truth

        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1, youden_index)
        """
        # Filter results with ground truth
        valid_results = [r for r in results if r.mapping.ground_truth is not None]

        if not valid_results:
            return {}

        # Calculate confusion matrix
        tp = sum(1 for r in valid_results if r.prediction and r.mapping.ground_truth)
        tn = sum(1 for r in valid_results if not r.prediction and not r.mapping.ground_truth)
        fp = sum(1 for r in valid_results if r.prediction and not r.mapping.ground_truth)
        fn = sum(1 for r in valid_results if not r.prediction and r.mapping.ground_truth)

        total = len(valid_results)
        accuracy = (tp + tn) / total if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Youden's Index (as mentioned in instructions.md)
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_index = sensitivity + specificity - 1

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "youden_index": youden_index,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "total": total
        }
