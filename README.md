# LLM Oracle for Ontology Alignment

PNLF^S (Natural-language Friendly with Synonyms) implementation for ontology alignment using LLM-based oracle.

## Overview

This project implements an LLM-based oracle diagnostic engine for ontology alignment, as specified in `instructions.md`. The system uses the **PNLF^S** prompt template to diagnose whether two ontology entities from different ontologies represent the same concept.

### Key Features

- **PNLF^S Prompt Builder**: Implements the optimal prompt template for ontology alignment
- **Multi-Provider Support**: Works with OpenAI GPT, Anthropic Claude, and Google Gemini (recommended)
- **Interactive Learning Environment**: CLI tool for experimenting with prompts and diagnostics
- **Batch Evaluation**: Evaluate performance on multiple mappings with standard metrics
- **Sample Dataset**: 10 biomedical ontology mapping examples included

## Project Structure

```
ccw-ontology_aligment_with_LLM/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Data models for entities and mappings
│   ├── prompt_builder.py     # PNLF^S prompt builder implementation
│   └── oracle.py             # LLM oracle diagnostic engine
├── examples/
│   └── sample_mappings.json  # Sample biomedical ontology mappings
├── interactive.py            # Interactive CLI learning environment
├── requirements.txt          # Python dependencies
├── instructions.md           # Original specification (PNLF^S)
└── README.md                # This file
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ccw-ontology_aligment_with_LLM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: You only need to install the LLM provider(s) you plan to use:

- **OpenAI**: `pip install openai`
- **Anthropic**: `pip install anthropic`
- **Google Gemini** (recommended): `pip install google-generativeai`

### 3. Set up API keys

Set the appropriate environment variable for your chosen provider:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# For Google Gemini (recommended)
export GOOGLE_API_KEY="your-api-key"
```

## Usage

### Interactive Learning Environment

The easiest way to explore the system is through the interactive CLI:

```bash
python interactive.py
```

#### Features:

1. **Preview PNLF^S Prompt**: See how prompts are generated without calling the LLM
2. **Diagnose with LLM Oracle**: Send prompts to your configured LLM and get True/False predictions
3. **Batch Evaluation**: Evaluate all sample mappings and see performance metrics
4. **Manual Input Mode**: Enter your own entity pairs for diagnosis
5. **Configure LLM Provider**: Choose and configure OpenAI, Anthropic, Gemini, or Mock provider
6. **View Sample Mappings**: Browse the included biomedical ontology examples
7. **About PNLF^S**: Learn about the prompt template design

### Programmatic Usage

#### Basic Example

```python
from src.models import OntologyEntity, EntityMapping
from src.oracle import OntologyAlignmentOracle, GeminiProvider

# Create entities
source = OntologyEntity(
    label="alveolus epithelium",
    parent_class="lung epithelium"
)

target = OntologyEntity(
    label="Alveolar_Epithelium",
    parent_class="Epithelium",
    synonyms=["Lung Alveolar Epithelia", "Alveolar Epithelium"]
)

# Create mapping
mapping = EntityMapping(source_entity=source, target_entity=target)

# Initialize oracle with Gemini Flash 2.5 (recommended)
provider = GeminiProvider(model="gemini-2.0-flash-exp")
oracle = OntologyAlignmentOracle(
    provider=provider,
    system_prompt_type="biomedical_specialist"
)

# Diagnose
result = oracle.diagnose_mapping(mapping)
print(f"Prediction: {result.prediction}")  # True or False
```

#### Batch Evaluation

```python
from src.oracle import OntologyAlignmentOracle, GeminiProvider

# Load your mappings
mappings = [...]  # List of EntityMapping objects

# Initialize oracle
provider = GeminiProvider()
oracle = OntologyAlignmentOracle(provider)

# Diagnose all mappings
results = oracle.diagnose_batch(mappings)

# Evaluate performance
metrics = oracle.evaluate_performance(results)
print(f"Youden's Index: {metrics['youden_index']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

#### Custom Prompt Builder

```python
from src.prompt_builder import PNLFSPromptBuilder

# Build prompt directly from attributes
prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    source_label="cardiac muscle",
    source_parent="muscle tissue",
    target_label="heart muscle",
    target_parent="muscle tissue",
    target_synonyms=["myocardium", "cardiac muscle"]
)

print(prompt)
```

## PNLF^S Prompt Template

The **PNLF^S** (Natural-language Friendly with Synonyms) template is the optimal prompt design for LLM-based ontology alignment, achieving Youden's Index > 0.5 on uncertain mappings.

### Template Structure

```
We have two entities from different ontologies.
The first one is "[source_label]", which falls under the category "[source_parent]".
The second one is "[target_label]", also known as "[synonym1]", "[synonym2]", ..., which falls under the category "[target_parent]".
Do they mean the same thing? Respond with "True" or "False".
```

### Example

```
We have two entities from different ontologies.
The first one is "alveolus epithelium", which falls under the category "lung epithelium".
The second one is "Alveolar_Epithelium", also known as "Lung Alveolar Epithelia", "Alveolar Epithelium", "Epithelia of lung alveoli", which falls under the category "Epithelium".
Do they mean the same thing? Respond with "True" or "False".
```

### Design Principles

1. **NLF (Natural-language Friendly)**: Uses natural human language to leverage LLM's training on large corpora
2. **S (with Synonyms)**: Explicitly includes entity synonyms to help LLM interpret vocabulary variations
3. **Structured Output**: Enforces boolean (True/False) responses to prevent hallucination

## Recommended LLM Model

According to the research in `instructions.md`, the recommended model is:

- **Google Gemini Flash 2.5**: Achieves average Youden's Index of 0.550
- **Alternatives**: GPT-4o Mini, Gemini Flash 2.0 (good cost-performance balance)

## Sample Mappings

The `examples/sample_mappings.json` file contains 10 biomedical ontology mapping examples:

1. Mouse-Human Alveolus Epithelium (from instructions.md)
2. Cardiac Muscle vs Skeletal Muscle (False)
3. Neuron vs Nerve Cell (True)
4. Hepatocyte vs Liver Cell (True)
5. Erythrocyte vs Leukocyte (False)
6. Pulmonary Artery vs Lung Artery (True)
7. Kidney vs Renal Organ (True)
8. Pancreas vs Spleen (False)
9. Myocardium vs Heart Muscle (True)
10. Dendrite vs Axon (False)

## Evaluation Metrics

The system calculates standard performance metrics:

- **Youden's Index**: Sensitivity + Specificity - 1 (target > 0.5)
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall

## System Prompts (Optional)

Three system prompt options are available:

1. **Biomedical Specialist**: Positions LLM as a biomedical ontology expert
2. **Explicit Semantics Focus**: Instructs LLM to focus on provided synonyms and parent classes
3. **Combined**: Combines both approaches

**Note**: According to `instructions.md`, system prompts provide marginal improvements but help with consistency.

## Development and Testing

### Running Tests (Manual)

You can test the prompt builder without API calls:

```bash
python -c "
from src.prompt_builder import PNLFSPromptBuilder
prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    'neuron', 'nerve cell',
    'nerve cell', 'nervous system cell',
    ['neuron', 'neurocyte']
)
print(prompt)
"
```

### Mock Provider for Testing

Use the `MockProvider` to test without making API calls:

```python
from src.oracle import OntologyAlignmentOracle, MockProvider

oracle = OntologyAlignmentOracle(MockProvider(default_response=True))
result = oracle.diagnose_mapping(mapping)
```

## Citation

If you use this implementation, please cite the original research paper that specifies the PNLF^S approach (see `instructions.md` for details).

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- `instructions.md`: Full specification of the LLM Oracle diagnostic engine
- PNLF^S: Natural-language Friendly with Synonyms prompt template
- Youden's Index: Statistical measure of diagnostic performance

## Support

For questions or issues, please open an issue on the repository.
