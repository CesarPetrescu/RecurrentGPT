"""
Hierarchical Memory System for RecurrentGPT

This module implements a structured memory system that maintains:
- Document outline tree
- Main claims per section with supporting evidence
- Hierarchical context retrieval
- Logical flow tracking
"""

from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Claim:
    """Represents a main claim or assertion made in the document"""
    text: str
    evidence: List[str] = field(default_factory=list)
    section_id: str = ""
    paragraph_index: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Section:
    """Represents a section in the document outline"""
    id: str
    title: str
    content: str = ""
    claims: List[Claim] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    paragraph_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['claims'] = [c.to_dict() for c in self.claims]
        return data


class HierarchicalMemory:
    """
    Manages document structure as a tree with claims and evidence.

    This replaces the flat list approach with:
    - Outline tree structure
    - Claim-evidence mapping
    - Hierarchical context retrieval
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.sections: Dict[str, Section] = {}
        self.paragraphs: List[str] = []
        self.root_section_ids: List[str] = []
        self.current_section_id: Optional[str] = None
        self.section_counter = 0

    def create_section(self, title: str, parent_id: Optional[str] = None) -> str:
        """Create a new section in the outline tree"""
        section_id = f"section_{self.section_counter}"
        self.section_counter += 1

        depth = 0
        if parent_id and parent_id in self.sections:
            parent = self.sections[parent_id]
            parent.children_ids.append(section_id)
            depth = parent.depth + 1
        else:
            self.root_section_ids.append(section_id)

        section = Section(
            id=section_id,
            title=title,
            parent_id=parent_id,
            depth=depth
        )
        self.sections[section_id] = section
        return section_id

    def add_paragraph(self, section_id: str, paragraph: str, claims: List[Dict] = None):
        """Add a paragraph to a section with optional extracted claims"""
        if section_id not in self.sections:
            raise ValueError(f"Section {section_id} does not exist")

        paragraph_index = len(self.paragraphs)
        self.paragraphs.append(paragraph)

        section = self.sections[section_id]
        section.paragraph_indices.append(paragraph_index)
        section.content += "\n\n" + paragraph if section.content else paragraph

        # Add claims if provided
        if claims:
            for claim_data in claims:
                claim = Claim(
                    text=claim_data.get('text', ''),
                    evidence=claim_data.get('evidence', []),
                    section_id=section_id,
                    paragraph_index=paragraph_index
                )
                section.claims.append(claim)

    def get_section_context(self, section_id: str, include_ancestors: bool = True,
                           include_siblings: bool = False) -> str:
        """
        Get contextual information for a section.

        Args:
            section_id: Target section
            include_ancestors: Include parent and ancestor sections
            include_siblings: Include sibling sections

        Returns:
            Formatted context string
        """
        if section_id not in self.sections:
            return ""

        context_parts = []
        section = self.sections[section_id]

        # Add ancestor context (provides hierarchical awareness)
        if include_ancestors and section.parent_id:
            ancestor_chain = self._get_ancestor_chain(section_id)
            for ancestor_id in reversed(ancestor_chain):
                ancestor = self.sections[ancestor_id]
                context_parts.append(f"# {ancestor.title}")
                if ancestor.claims:
                    context_parts.append("Main points:")
                    for claim in ancestor.claims[:3]:  # Limit to top 3 claims
                        context_parts.append(f"- {claim.text}")

        # Add sibling context (provides lateral awareness)
        if include_siblings and section.parent_id:
            siblings = self.sections[section.parent_id].children_ids
            for sibling_id in siblings:
                if sibling_id != section_id:
                    sibling = self.sections[sibling_id]
                    context_parts.append(f"## {sibling.title}")
                    if sibling.claims:
                        context_parts.append(f"- {sibling.claims[0].text}")

        return "\n".join(context_parts)

    def get_relevant_claims(self, query: str, top_k: int = 5) -> List[Tuple[Claim, float]]:
        """
        Retrieve claims most relevant to a query using embeddings.

        Returns list of (Claim, similarity_score) tuples.
        """
        if not self.embedder:
            return []

        all_claims = []
        for section in self.sections.values():
            all_claims.extend(section.claims)

        if not all_claims:
            return []

        # Embed query and all claims
        claim_texts = [c.text for c in all_claims]
        claim_embeddings = self.embedder.encode(claim_texts)
        query_embedding = self.embedder.encode([query])[0]

        # Compute similarities
        scores = self.embedder.cosine_scores(query_embedding, claim_embeddings)

        # Get top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(all_claims[i], scores[i]) for i in top_indices]

    def get_relevant_paragraphs(self, query: str, top_k: int = 3,
                                section_id: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """
        Retrieve paragraphs most relevant to a query.

        Args:
            query: Search query
            top_k: Number of results
            section_id: If provided, prioritize this section and its ancestors

        Returns:
            List of (paragraph_text, similarity_score, section_title) tuples
        """
        if not self.embedder or not self.paragraphs:
            return []

        # Get embeddings
        paragraph_embeddings = self.embedder.encode(self.paragraphs)
        query_embedding = self.embedder.encode([query])[0]
        scores = self.embedder.cosine_scores(query_embedding, paragraph_embeddings)

        # Build paragraph metadata
        paragraph_meta = []
        for idx, para in enumerate(self.paragraphs):
            # Find which section this paragraph belongs to
            section_title = "Unknown"
            is_priority = False

            for sec in self.sections.values():
                if idx in sec.paragraph_indices:
                    section_title = sec.title
                    # Check if this section is in the priority chain
                    if section_id:
                        ancestor_chain = self._get_ancestor_chain(section_id)
                        if sec.id == section_id or sec.id in ancestor_chain:
                            is_priority = True
                    break

            paragraph_meta.append({
                'text': para,
                'score': scores[idx],
                'section': section_title,
                'is_priority': is_priority
            })

        # Boost scores for priority sections
        if section_id:
            for meta in paragraph_meta:
                if meta['is_priority']:
                    meta['score'] *= 1.3  # 30% boost for hierarchically related content

        # Sort by score and take top-k
        sorted_meta = sorted(paragraph_meta, key=lambda x: x['score'], reverse=True)[:top_k]

        return [(m['text'], m['score'], m['section']) for m in sorted_meta]

    def _get_ancestor_chain(self, section_id: str) -> List[str]:
        """Get list of ancestor section IDs from root to parent"""
        chain = []
        current_id = section_id

        while current_id and current_id in self.sections:
            section = self.sections[current_id]
            if section.parent_id:
                chain.append(section.parent_id)
                current_id = section.parent_id
            else:
                break

        return chain

    def get_outline_tree(self, max_depth: int = -1) -> str:
        """
        Generate a text representation of the outline tree.

        Args:
            max_depth: Maximum depth to display (-1 for unlimited)

        Returns:
            Formatted outline string
        """
        lines = []

        def traverse(section_id: str, indent: int = 0):
            if max_depth >= 0 and indent > max_depth:
                return

            section = self.sections[section_id]
            prefix = "  " * indent + ("- " if indent > 0 else "")
            lines.append(f"{prefix}{section.title}")

            # Add claim count
            if section.claims:
                lines.append(f"{' ' * (len(prefix) + 2)}[{len(section.claims)} claims]")

            # Recurse to children
            for child_id in section.children_ids:
                traverse(child_id, indent + 1)

        for root_id in self.root_section_ids:
            traverse(root_id)

        return "\n".join(lines)

    def get_document_summary(self) -> Dict:
        """Get a summary of the document structure"""
        return {
            'total_sections': len(self.sections),
            'total_paragraphs': len(self.paragraphs),
            'total_claims': sum(len(s.claims) for s in self.sections.values()),
            'max_depth': max((s.depth for s in self.sections.values()), default=0),
            'outline': self.get_outline_tree()
        }

    def export_to_dict(self) -> Dict:
        """Export complete memory state to dictionary"""
        return {
            'sections': {sid: s.to_dict() for sid, s in self.sections.items()},
            'paragraphs': self.paragraphs,
            'root_section_ids': self.root_section_ids,
            'current_section_id': self.current_section_id,
            'section_counter': self.section_counter
        }

    def import_from_dict(self, data: Dict):
        """Import memory state from dictionary"""
        self.paragraphs = data['paragraphs']
        self.root_section_ids = data['root_section_ids']
        self.current_section_id = data.get('current_section_id')
        self.section_counter = data['section_counter']

        # Reconstruct sections
        self.sections = {}
        for sid, sdata in data['sections'].items():
            claims = [Claim(**c) for c in sdata['claims']]
            sdata['claims'] = claims
            self.sections[sid] = Section(**sdata)
