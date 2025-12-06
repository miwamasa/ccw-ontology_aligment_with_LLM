#!/usr/bin/env python3
"""
Demo script to showcase the PNLF^S prompt builder and LLM oracle
without requiring API keys (uses mock provider)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import OntologyEntity, EntityMapping
from src.prompt_builder import PNLFSPromptBuilder, SYSTEM_PROMPTS
from src.oracle import OntologyAlignmentOracle, MockProvider


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_prompt_builder():
    """Demonstrate the PNLF^S prompt builder"""
    print_section("DEMO 1: PNLF^S Prompt Builder")

    print("=== Biomedical Example (from instructions.md) ===\n")

    # Example from instructions.md
    source = OntologyEntity(
        label="alveolus epithelium",
        parent_class="lung epithelium",
        ontology_id="mouse:MA_0001771"
    )

    target = OntologyEntity(
        label="Alveolar_Epithelium",
        parent_class="Epithelium",
        synonyms=[
            "Lung Alveolar Epithelia",
            "Alveolar Epithelium",
            "Epithelia of lung alveoli"
        ],
        ontology_id="human:NCI_C12867"
    )

    mapping = EntityMapping(
        source_entity=source,
        target_entity=target,
        ground_truth=True
    )

    print("Input Entities:")
    print(f"  Source: {source.label} (parent: {source.parent_class})")
    print(f"  Target: {target.label} (parent: {target.parent_class})")
    print(f"  Synonyms: {', '.join(target.synonyms)}")

    print("\nGenerated PNLF^S Prompt:")
    print("-" * 70)
    prompt = PNLFSPromptBuilder.build_prompt(mapping)
    print(prompt)
    print("-" * 70)

    print("\n=== Manufacturing/IoT Example ===\n")

    # Manufacturing/IoT example
    iot_source = OntologyEntity(
        label="temperature sensor",
        parent_class="environmental sensor"
    )

    iot_target = OntologyEntity(
        label="thermometer",
        parent_class="measuring instrument",
        synonyms=["temperature sensor", "temperature measuring device", "thermal detector"]
    )

    iot_mapping = EntityMapping(
        source_entity=iot_source,
        target_entity=iot_target,
        ground_truth=True
    )

    print("Input Entities:")
    print(f"  Source: {iot_source.label} (parent: {iot_source.parent_class})")
    print(f"  Target: {iot_target.label} (parent: {iot_target.parent_class})")
    print(f"  Synonyms: {', '.join(iot_target.synonyms)}")

    print("\nGenerated PNLF^S Prompt:")
    print("-" * 70)
    iot_prompt = PNLFSPromptBuilder.build_prompt(iot_mapping)
    print(iot_prompt)
    print("-" * 70)

    print("\nThis prompt will be sent to the LLM for True/False diagnosis.")


def demo_direct_builder():
    """Demonstrate direct prompt building without entity objects"""
    print_section("DEMO 2: Direct Prompt Building")

    print("Building prompt directly from parameters:\n")

    prompt = PNLFSPromptBuilder.build_prompt_from_entities(
        source_label="hepatocyte",
        source_parent="epithelial cell",
        target_label="liver cell",
        target_parent="organ cell",
        target_synonyms=["hepatocyte", "hepatic parenchymal cell"]
    )

    print(prompt)


def demo_mock_oracle():
    """Demonstrate mock oracle (no API required)"""
    print_section("DEMO 3: Mock LLM Oracle (No API Required)")

    # Create sample mappings
    mappings = [
        EntityMapping(
            source_entity=OntologyEntity("neuron", "nerve cell"),
            target_entity=OntologyEntity(
                "nerve cell",
                "nervous system cell",
                synonyms=["neuron", "neurocyte"]
            ),
            ground_truth=True
        ),
        EntityMapping(
            source_entity=OntologyEntity("erythrocyte", "blood cell", ["red blood cell"]),
            target_entity=OntologyEntity(
                "leukocyte",
                "blood cell",
                synonyms=["white blood cell"]
            ),
            ground_truth=False
        ),
    ]

    # Initialize mock oracle
    oracle = OntologyAlignmentOracle(
        provider=MockProvider(default_response=True),
        system_prompt_type="biomedical_specialist"
    )

    print("Mock Provider Configuration: Always returns True (for testing)\n")

    # Diagnose each mapping
    for i, mapping in enumerate(mappings, 1):
        print(f"Mapping {i}:")
        print(f"  {mapping.source_entity.label} <-> {mapping.target_entity.label}")
        print(f"  Ground Truth: {mapping.ground_truth}")

        result = oracle.diagnose_mapping(mapping)
        correct = "✓" if result.is_correct else "✗"

        print(f"  Prediction: {result.prediction} {correct}")
        print()


def demo_system_prompts():
    """Show available system prompts"""
    print_section("DEMO 4: System Prompts (Optional)")

    print("Available system prompts:\n")

    for name, prompt in SYSTEM_PROMPTS.items():
        print(f"{name}:")
        print(f"  {prompt}\n")


def demo_batch_evaluation():
    """Demonstrate batch evaluation with metrics"""
    print_section("DEMO 5: Batch Evaluation with Metrics")

    # Create test mappings with ground truth
    test_mappings = [
        EntityMapping(
            OntologyEntity("kidney", "excretory organ"),
            OntologyEntity("renal organ", "urinary system organ", ["kidney"]),
            ground_truth=True
        ),
        EntityMapping(
            OntologyEntity("pancreas", "digestive gland"),
            OntologyEntity("spleen", "lymphoid organ", ["splenic organ"]),
            ground_truth=False
        ),
        EntityMapping(
            OntologyEntity("myocardium", "cardiac tissue"),
            OntologyEntity("heart muscle", "muscle tissue", ["myocardium"]),
            ground_truth=True
        ),
        EntityMapping(
            OntologyEntity("dendrite", "neuron projection"),
            OntologyEntity("axon", "neuron projection", ["nerve fiber"]),
            ground_truth=False
        ),
    ]

    # Mock oracle with controlled responses
    oracle = OntologyAlignmentOracle(MockProvider(default_response=True))

    print("Running batch evaluation on 4 test mappings...\n")

    results = oracle.diagnose_batch(test_mappings)

    # Display individual results
    for i, result in enumerate(results, 1):
        mapping = result.mapping
        correct = "✓" if result.is_correct else "✗"
        print(f"{i}. {mapping.source_entity.label} <-> {mapping.target_entity.label}")
        print(f"   Prediction: {result.prediction}, Truth: {mapping.ground_truth} {correct}")

    # Calculate and display metrics
    metrics = oracle.evaluate_performance(results)

    print("\nPerformance Metrics:")
    print("-" * 70)
    print(f"Accuracy:       {metrics['accuracy']:.3f}")
    print(f"Precision:      {metrics['precision']:.3f}")
    print(f"Recall:         {metrics['recall']:.3f}")
    print(f"F1 Score:       {metrics['f1_score']:.3f}")
    print(f"Youden's Index: {metrics['youden_index']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("-" * 70)

    print("\nNote: Mock provider always returns True, so results are for demonstration only.")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  LLM Oracle for Ontology Alignment - Demo")
    print("  PNLF^S Implementation Showcase")
    print("="*70)

    try:
        demo_prompt_builder()
        input("\n[Press Enter to continue...]")

        demo_direct_builder()
        input("\n[Press Enter to continue...]")

        demo_mock_oracle()
        input("\n[Press Enter to continue...]")

        demo_system_prompts()
        input("\n[Press Enter to continue...]")

        demo_batch_evaluation()

        print_section("Demo Complete!")
        print("To try the interactive environment, run:")
        print("  python interactive.py")
        print("\nTo use with real LLM providers, set up your API keys and configure")
        print("the provider in interactive.py or use the programmatic API.")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
