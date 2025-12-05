#!/usr/bin/env python3
"""
Interactive CLI for LLM Oracle Learning Environment
Allows users to experiment with PNLF^S prompts and LLM diagnosis
"""

import os
import sys
import json
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import OntologyEntity, EntityMapping
from src.prompt_builder import PNLFSPromptBuilder, SYSTEM_PROMPTS
from src.oracle import (
    OntologyAlignmentOracle,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    MockProvider
)


class InteractiveCLI:
    """Interactive CLI for LLM Oracle experimentation"""

    def __init__(self):
        self.oracle = None
        self.sample_mappings = []
        self.load_sample_mappings()

    def load_sample_mappings(self):
        """Load sample mappings from examples directory"""
        sample_file = os.path.join(
            os.path.dirname(__file__),
            'examples',
            'sample_mappings.json'
        )

        if os.path.exists(sample_file):
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    source = OntologyEntity(
                        label=item['source']['label'],
                        parent_class=item['source']['parent_class'],
                        synonyms=item['source'].get('synonyms', []),
                        ontology_id=item['source'].get('ontology_id')
                    )
                    target = OntologyEntity(
                        label=item['target']['label'],
                        parent_class=item['target']['parent_class'],
                        synonyms=item['target'].get('synonyms', []),
                        ontology_id=item['target'].get('ontology_id')
                    )
                    mapping = EntityMapping(
                        source_entity=source,
                        target_entity=target,
                        ground_truth=item.get('ground_truth')
                    )
                    self.sample_mappings.append({
                        'name': item.get('name', 'Unknown'),
                        'description': item.get('description', ''),
                        'mapping': mapping
                    })

    def print_header(self):
        """Print application header"""
        print("\n" + "="*70)
        print("  LLM Oracle for Ontology Alignment - Interactive Learning Environment")
        print("  PNLF^S (Natural-language Friendly with Synonyms) Implementation")
        print("="*70 + "\n")

    def print_menu(self):
        """Print main menu"""
        print("\n--- Main Menu ---")
        print("1. Preview PNLF^S Prompt")
        print("2. Diagnose with LLM Oracle")
        print("3. Batch Evaluation on Samples")
        print("4. Manual Input Mode")
        print("5. Configure LLM Provider")
        print("6. View Sample Mappings")
        print("7. About PNLF^S")
        print("0. Exit")
        print()

    def preview_prompt_mode(self):
        """Preview PNLF^S prompt without calling LLM"""
        print("\n--- Preview PNLF^S Prompt ---\n")

        mapping = self.select_mapping()
        if not mapping:
            return

        prompt = PNLFSPromptBuilder.build_prompt(mapping)

        print("\n" + "-"*70)
        print("Generated PNLF^S Prompt:")
        print("-"*70)
        print(prompt)
        print("-"*70)

        # Show which system prompt would be used
        if self.oracle and self.oracle.system_prompt:
            print("\nSystem Prompt:")
            print("-"*70)
            print(self.oracle.system_prompt)
            print("-"*70)

    def diagnose_mode(self):
        """Diagnose a mapping with LLM oracle"""
        print("\n--- Diagnose with LLM Oracle ---\n")

        if not self.oracle:
            print("⚠ No LLM provider configured. Using Mock provider.")
            self.oracle = OntologyAlignmentOracle(MockProvider())

        mapping = self.select_mapping()
        if not mapping:
            return

        # Show prompt
        prompt = PNLFSPromptBuilder.build_prompt(mapping)
        print("\nPrompt to be sent:")
        print("-"*70)
        print(prompt)
        print("-"*70)

        # Confirm
        confirm = input("\nSend to LLM? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

        # Diagnose
        print("\nDiagnosing...")
        try:
            result = self.oracle.diagnose_mapping(mapping)

            print("\n" + "="*70)
            print("Diagnostic Result:")
            print("="*70)
            print(f"Prediction: {result.prediction}")
            if mapping.ground_truth is not None:
                print(f"Ground Truth: {mapping.ground_truth}")
                print(f"Correct: {result.is_correct}")
            print("="*70)

        except Exception as e:
            print(f"\n❌ Error during diagnosis: {e}")

    def batch_evaluation_mode(self):
        """Run batch evaluation on all sample mappings"""
        print("\n--- Batch Evaluation ---\n")

        if not self.oracle:
            print("⚠ No LLM provider configured. Using Mock provider.")
            self.oracle = OntologyAlignmentOracle(MockProvider())

        if not self.sample_mappings:
            print("No sample mappings available.")
            return

        print(f"Running evaluation on {len(self.sample_mappings)} sample mappings...")

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

        # Run diagnostics
        mappings = [item['mapping'] for item in self.sample_mappings]

        print("\nProcessing...")
        try:
            results = self.oracle.diagnose_batch(mappings)

            # Display results
            print("\n" + "="*70)
            print("Batch Evaluation Results")
            print("="*70)

            for i, (result, item) in enumerate(zip(results, self.sample_mappings), 1):
                print(f"\n{i}. {item['name']}")
                print(f"   Prediction: {result.prediction}")
                if result.mapping.ground_truth is not None:
                    correct = "✓" if result.is_correct else "✗"
                    print(f"   Ground Truth: {result.mapping.ground_truth} {correct}")

            # Calculate metrics
            metrics = self.oracle.evaluate_performance(results)

            if metrics:
                print("\n" + "="*70)
                print("Performance Metrics")
                print("="*70)
                print(f"Accuracy:       {metrics['accuracy']:.3f}")
                print(f"Precision:      {metrics['precision']:.3f}")
                print(f"Recall:         {metrics['recall']:.3f}")
                print(f"F1 Score:       {metrics['f1_score']:.3f}")
                print(f"Youden's Index: {metrics['youden_index']:.3f}")
                print(f"\nConfusion Matrix:")
                print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}")
                print(f"  FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}")
                print("="*70)

        except Exception as e:
            print(f"\n❌ Error during batch evaluation: {e}")

    def manual_input_mode(self):
        """Manual input mode for custom entity pairs"""
        print("\n--- Manual Input Mode ---\n")

        print("Enter information for two ontology entities:\n")

        # Source entity
        print("Source Entity:")
        source_label = input("  Label: ").strip()
        source_parent = input("  Parent class: ").strip()

        # Target entity
        print("\nTarget Entity:")
        target_label = input("  Label: ").strip()
        target_parent = input("  Parent class: ").strip()
        target_synonyms_str = input("  Synonyms (comma-separated): ").strip()
        target_synonyms = [s.strip() for s in target_synonyms_str.split(',') if s.strip()]

        # Create mapping
        source = OntologyEntity(label=source_label, parent_class=source_parent)
        target = OntologyEntity(
            label=target_label,
            parent_class=target_parent,
            synonyms=target_synonyms
        )
        mapping = EntityMapping(source_entity=source, target_entity=target)

        # Preview prompt
        prompt = PNLFSPromptBuilder.build_prompt(mapping)
        print("\n" + "-"*70)
        print("Generated PNLF^S Prompt:")
        print("-"*70)
        print(prompt)
        print("-"*70)

        # Ask if they want to diagnose
        if not self.oracle:
            print("\n⚠ No LLM provider configured.")
            return

        diagnose = input("\nDiagnose with LLM? (y/n): ").strip().lower()
        if diagnose == 'y':
            print("\nDiagnosing...")
            try:
                result = self.oracle.diagnose_mapping(mapping)
                print(f"\nPrediction: {result.prediction}")
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def configure_provider_mode(self):
        """Configure LLM provider"""
        print("\n--- Configure LLM Provider ---\n")

        print("Available providers:")
        print("1. Mock (no API calls, for testing)")
        print("2. OpenAI (GPT-4o, GPT-4o Mini)")
        print("3. Anthropic (Claude)")
        print("4. Google Gemini (Recommended: Flash 2.5)")
        print()

        choice = input("Select provider (1-4): ").strip()

        system_prompt_type = self.select_system_prompt()

        try:
            if choice == '1':
                self.oracle = OntologyAlignmentOracle(
                    MockProvider(),
                    system_prompt_type=system_prompt_type
                )
                print("\n✓ Mock provider configured")

            elif choice == '2':
                model = input("Model name (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
                provider = OpenAIProvider(model=model)
                self.oracle = OntologyAlignmentOracle(
                    provider,
                    system_prompt_type=system_prompt_type
                )
                print(f"\n✓ OpenAI provider configured with model: {model}")

            elif choice == '3':
                model = input("Model name (default: claude-3-5-sonnet-20241022): ").strip() or "claude-3-5-sonnet-20241022"
                provider = AnthropicProvider(model=model)
                self.oracle = OntologyAlignmentOracle(
                    provider,
                    system_prompt_type=system_prompt_type
                )
                print(f"\n✓ Anthropic provider configured with model: {model}")

            elif choice == '4':
                model = input("Model name (default: gemini-2.0-flash-exp): ").strip() or "gemini-2.0-flash-exp"
                provider = GeminiProvider(model=model)
                self.oracle = OntologyAlignmentOracle(
                    provider,
                    system_prompt_type=system_prompt_type
                )
                print(f"\n✓ Google Gemini provider configured with model: {model}")

            else:
                print("Invalid choice.")

        except Exception as e:
            print(f"\n❌ Configuration error: {e}")
            print("Make sure you have set the appropriate API key environment variable.")

    def select_system_prompt(self) -> Optional[str]:
        """Select system prompt type"""
        print("\nSystem Prompt:")
        print("1. None")
        print("2. Biomedical Specialist")
        print("3. Explicit Semantics Focus")
        print("4. Combined")
        print()

        choice = input("Select system prompt (1-4, default: 1): ").strip() or "1"

        if choice == '2':
            return "biomedical_specialist"
        elif choice == '3':
            return "explicit_semantics"
        elif choice == '4':
            return "combined"
        else:
            return None

    def view_samples_mode(self):
        """View all sample mappings"""
        print("\n--- Sample Mappings ---\n")

        if not self.sample_mappings:
            print("No sample mappings available.")
            return

        for i, item in enumerate(self.sample_mappings, 1):
            mapping = item['mapping']
            print(f"{i}. {item['name']}")
            print(f"   {item['description']}")
            print(f"   Source: {mapping.source_entity.label} ({mapping.source_entity.parent_class})")
            print(f"   Target: {mapping.target_entity.label} ({mapping.target_entity.parent_class})")
            if mapping.target_entity.synonyms:
                print(f"   Synonyms: {', '.join(mapping.target_entity.synonyms)}")
            if mapping.ground_truth is not None:
                print(f"   Ground Truth: {mapping.ground_truth}")
            print()

    def about_pnlfs_mode(self):
        """Display information about PNLF^S"""
        print("\n" + "="*70)
        print("About PNLF^S (Natural-language Friendly with Synonyms)")
        print("="*70)
        print("""
PNLF^S is the optimal prompt template for LLM-based ontology alignment
oracle, as specified in instructions.md.

Design Principles:
1. NLF (Natural-language Friendly): Uses natural language to leverage
   LLM's training on large corpora
2. S (with Synonyms): Explicitly includes entity synonyms to help LLM
   interpret vocabulary variations

Template Structure:
1. Introduction: "We have two entities from different ontologies."
2. Source entity: "[label]" under category "[parent class]"
3. Target entity: "[label]", also known as "[synonyms]",
                  under category "[parent class]"
4. Question: "Do they mean the same thing? True or False."

Performance:
- Achieves Youden's Index > 0.5 on uncertain mappings (M_ask)
- Recommended LLM: Gemini Flash 2.5 (average YI: 0.550)
- Alternative: GPT-4o Mini, Gemini Flash 2.0

Reference: See instructions.md for full specification
        """)
        print("="*70)

    def select_mapping(self) -> Optional[EntityMapping]:
        """Select a mapping from samples"""
        if not self.sample_mappings:
            print("No sample mappings available.")
            return None

        print("\nAvailable mappings:")
        for i, item in enumerate(self.sample_mappings, 1):
            print(f"{i}. {item['name']}")

        print()
        choice = input(f"Select mapping (1-{len(self.sample_mappings)}): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.sample_mappings):
                item = self.sample_mappings[idx]
                print(f"\nSelected: {item['name']}")
                print(f"{item['description']}")
                return item['mapping']
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None

    def run(self):
        """Main loop"""
        self.print_header()

        while True:
            self.print_menu()
            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.preview_prompt_mode()
            elif choice == '2':
                self.diagnose_mode()
            elif choice == '3':
                self.batch_evaluation_mode()
            elif choice == '4':
                self.manual_input_mode()
            elif choice == '5':
                self.configure_provider_mode()
            elif choice == '6':
                self.view_samples_mode()
            elif choice == '7':
                self.about_pnlfs_mode()
            elif choice == '0':
                print("\nThank you for using LLM Oracle Learning Environment!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")


def main():
    """Entry point"""
    cli = InteractiveCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
