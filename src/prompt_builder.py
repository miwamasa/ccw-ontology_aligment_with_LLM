"""
PNLF^S Prompt Builder
Implements the PNLF^S (Natural-language Friendly with Synonyms) prompt template
as specified in instructions.md
"""

from typing import List
from .models import OntologyEntity, EntityMapping


class PNLFSPromptBuilder:
    """
    Builder for PNLF^S (Natural-language Friendly with Synonyms) prompts

    Template structure:
    1. NLF Frame: "We have two entities from different ontologies."
    2. Source e1: "The first one is '[label]', which falls under the category '[parent]'."
    3. Target e2: "The second one is '[label]', also known as '[synonyms]',
                   which falls under the category '[parent]'."
    4. Question: "Do they mean the same thing? Respond with 'True' or 'False'."
    """

    FRAME_INTRO = "We have two entities from different ontologies."
    SOURCE_TEMPLATE = 'The first one is "{label}", which falls under the category "{parent}".'
    TARGET_TEMPLATE = 'The second one is "{label}"{synonyms_clause}, which falls under the category "{parent}".'
    QUESTION = 'Do they mean the same thing? Respond with "True" or "False".'

    @staticmethod
    def _format_synonyms_clause(synonyms: List[str]) -> str:
        """
        Format the synonyms clause for the target entity

        Args:
            synonyms: List of synonym strings

        Returns:
            Formatted clause like ', also known as "syn1", "syn2", "syn3"'
            or empty string if no synonyms
        """
        if not synonyms:
            return ""

        # Format each synonym in quotes and join with commas
        formatted_synonyms = ', '.join(f'"{syn}"' for syn in synonyms)
        return f', also known as {formatted_synonyms}'

    @classmethod
    def build_prompt(cls, mapping: EntityMapping) -> str:
        """
        Build a PNLF^S prompt for the given entity mapping

        Args:
            mapping: EntityMapping containing source and target entities

        Returns:
            Formatted PNLF^S prompt string
        """
        source = mapping.source_entity
        target = mapping.target_entity

        # Build source entity description
        source_desc = cls.SOURCE_TEMPLATE.format(
            label=source.label,
            parent=source.parent_class
        )

        # Build target entity description with synonyms
        synonyms_clause = cls._format_synonyms_clause(target.synonyms)
        target_desc = cls.TARGET_TEMPLATE.format(
            label=target.label,
            synonyms_clause=synonyms_clause,
            parent=target.parent_class
        )

        # Combine all parts
        prompt = f"{cls.FRAME_INTRO}\n{source_desc}\n{target_desc}\n{cls.QUESTION}"

        return prompt

    @classmethod
    def build_prompt_from_entities(
        cls,
        source_label: str,
        source_parent: str,
        target_label: str,
        target_parent: str,
        target_synonyms: List[str] = None
    ) -> str:
        """
        Build a PNLF^S prompt directly from entity attributes

        Args:
            source_label: Label of source entity
            source_parent: Parent class of source entity
            target_label: Label of target entity
            target_parent: Parent class of target entity
            target_synonyms: List of target entity synonyms (optional)

        Returns:
            Formatted PNLF^S prompt string
        """
        source = OntologyEntity(
            label=source_label,
            parent_class=source_parent
        )

        target = OntologyEntity(
            label=target_label,
            parent_class=target_parent,
            synonyms=target_synonyms or []
        )

        mapping = EntityMapping(
            source_entity=source,
            target_entity=target
        )

        return cls.build_prompt(mapping)


# System prompts as specified in instructions.md (optional)
SYSTEM_PROMPTS = {
    "biomedical_specialist": (
        "You are a biomedical ontology specialist. "
        "Your task is to evaluate whether two entities from different ontologies "
        "represent the same concept. Focus on hierarchical and semantic context."
    ),
    "explicit_semantics": (
        "When evaluating entity mappings, rely primarily on the explicitly provided "
        "synonyms and parent class semantics. Consider the hierarchical context "
        "and vocabulary variations carefully."
    ),
    "combined": (
        "You are a biomedical ontology specialist evaluating entity mappings. "
        "When determining if two entities mean the same thing, focus on the "
        "explicitly provided synonyms and parent class semantics. "
        "Consider both hierarchical context and vocabulary variations."
    )
}
