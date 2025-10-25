"""
Multi-Level Planning System for RecurrentGPT

Maintains planning at multiple levels:
- High-level document outline (entire document)
- Mid-level section plans (next 3-5 sections)
- Low-level paragraph plans (immediate next paragraph)

This replaces the one-step-ahead planning with strategic document structure.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from utils import get_api_response, get_content_between_a_b


@dataclass
class SectionPlan:
    """Represents a planned section"""
    title: str
    purpose: str  # Why this section is needed
    key_points: List[str] = field(default_factory=list)
    estimated_paragraphs: int = 3
    status: str = "planned"  # planned, in_progress, completed
    dependencies: List[str] = field(default_factory=list)  # Sections that must come before

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentOutline:
    """Represents the high-level document structure"""
    title: str
    goal: str  # Overall goal of the document
    sections: List[SectionPlan] = field(default_factory=list)
    current_section_index: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data['sections'] = [s.to_dict() for s in self.sections]
        return data


class MultiLevelPlanner:
    """
    Manages planning at multiple levels of granularity.

    This enables the system to:
    - Maintain coherent document-level structure
    - Plan several sections ahead
    - Ensure each section serves the document's goals
    - Adapt plans based on what's been written
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.document_outline: Optional[DocumentOutline] = None
        self.upcoming_paragraph_plans: List[str] = []  # Next 3-5 paragraph plans

    def create_initial_outline(self, initial_input: str, num_sections: int = 10) -> DocumentOutline:
        """
        Create high-level document outline from initial input.

        Args:
            initial_input: The user's initial prompt/topic
            num_sections: Target number of major sections

        Returns:
            DocumentOutline with planned structure
        """
        prompt = f"""Create a comprehensive outline for a document on the following topic:

TOPIC/INITIAL INPUT:
{initial_input}

Create a structured outline with approximately {num_sections} major sections. For each section, specify:
1. Section title
2. Purpose (why this section is needed)
3. Key points to cover (3-5 bullet points)
4. Dependencies (which prior sections must be complete first)

Format your response as:

[Document Title]
<Proposed title for the complete document>

[Document Goal]
<1-2 sentences describing the overall goal>

[Section 1]
Title: <section title>
Purpose: <why this section is needed>
Key Points:
- <point 1>
- <point 2>
- <point 3>
Dependencies: <section titles this depends on, or "None">

[Section 2]
Title: <section title>
Purpose: <why this section is needed>
Key Points:
- <point 1>
- <point 2>
- <point 3>
Dependencies: <section titles this depends on, or "None">

... continue for all {num_sections} sections ...

Create a logical flow that builds knowledge progressively."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.7
        )

        try:
            doc_title = get_content_between_a_b('[Document Title]', '[Document Goal]', response).strip()
            doc_goal = get_content_between_a_b('[Document Goal]', '[Section 1]', response).strip()

            sections = []
            section_num = 1

            while f'[Section {section_num}]' in response:
                try:
                    # Determine the end marker
                    next_section = f'[Section {section_num + 1}]'
                    section_content = get_content_between_a_b(
                        f'[Section {section_num}]',
                        next_section if next_section in response else '',
                        response
                    )

                    # Parse section details
                    title = get_content_between_a_b('Title:', 'Purpose:', section_content).strip()
                    purpose = get_content_between_a_b('Purpose:', 'Key Points:', section_content).strip()
                    key_points_text = get_content_between_a_b('Key Points:', 'Dependencies:', section_content).strip()
                    dependencies_text = section_content.split('Dependencies:')[-1].strip()

                    # Parse key points
                    key_points = [
                        line.strip('- ').strip()
                        for line in key_points_text.split('\n')
                        if line.strip().startswith('-')
                    ]

                    # Parse dependencies
                    dependencies = []
                    if dependencies_text and dependencies_text.lower() != 'none':
                        dependencies = [d.strip() for d in dependencies_text.split(',')]

                    sections.append(SectionPlan(
                        title=title,
                        purpose=purpose,
                        key_points=key_points,
                        estimated_paragraphs=len(key_points),
                        dependencies=dependencies
                    ))

                    section_num += 1
                except Exception as e:
                    # Stop if we can't parse this section
                    break

            self.document_outline = DocumentOutline(
                title=doc_title,
                goal=doc_goal,
                sections=sections
            )

            return self.document_outline

        except Exception as e:
            # Fallback outline
            self.document_outline = DocumentOutline(
                title="Document",
                goal="Explore the topic systematically",
                sections=[
                    SectionPlan(
                        title="Introduction",
                        purpose="Introduce the topic",
                        key_points=["Define scope", "Outline structure"]
                    )
                ]
            )
            return self.document_outline

    def plan_next_paragraphs(self, current_section_index: int,
                            completed_paragraphs: List[str],
                            num_paragraphs: int = 3) -> List[str]:
        """
        Plan the next few paragraphs within the current section.

        Args:
            current_section_index: Index of current section in outline
            completed_paragraphs: Paragraphs written so far in this section
            num_paragraphs: Number of paragraphs to plan ahead

        Returns:
            List of paragraph-level instructions
        """
        if not self.document_outline or current_section_index >= len(self.document_outline.sections):
            return ["Continue developing the current topic."]

        current_section = self.document_outline.sections[current_section_index]
        completed_text = "\n\n".join(completed_paragraphs[-3:]) if completed_paragraphs else "None yet"

        prompt = f"""You are planning the next {num_paragraphs} paragraphs for a section of a document.

DOCUMENT GOAL: {self.document_outline.goal}

CURRENT SECTION:
Title: {current_section.title}
Purpose: {current_section.purpose}
Key Points to Cover:
{chr(10).join(['- ' + kp for kp in current_section.key_points])}

PARAGRAPHS COMPLETED IN THIS SECTION:
{completed_text}

Plan the next {num_paragraphs} paragraphs to advance this section toward completion. Each paragraph should:
1. Build on what's already written
2. Address remaining key points
3. Maintain logical flow
4. Move toward section conclusion if appropriate

Format your response as:

[Paragraph 1 Plan]
<Specific instruction for what this paragraph should accomplish>

[Paragraph 2 Plan]
<Specific instruction for what this paragraph should accomplish>

[Paragraph 3 Plan]
<Specific instruction for what this paragraph should accomplish>

Be specific about what each paragraph should cover."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.6
        )

        plans = []
        for i in range(1, num_paragraphs + 1):
            try:
                next_marker = f'[Paragraph {i+1} Plan]' if i < num_paragraphs else ''
                plan = get_content_between_a_b(
                    f'[Paragraph {i} Plan]',
                    next_marker,
                    response
                ).strip()
                plans.append(plan)
            except:
                # Fallback plan
                plans.append(f"Continue developing {current_section.title}")

        self.upcoming_paragraph_plans = plans
        return plans

    def advance_to_next_section(self) -> Optional[SectionPlan]:
        """
        Mark current section as complete and move to next.

        Returns:
            The next section plan, or None if document is complete
        """
        if not self.document_outline:
            return None

        current_idx = self.document_outline.current_section_index

        # Mark current section as completed
        if current_idx < len(self.document_outline.sections):
            self.document_outline.sections[current_idx].status = "completed"

        # Move to next section
        next_idx = current_idx + 1
        if next_idx < len(self.document_outline.sections):
            self.document_outline.current_section_index = next_idx
            self.document_outline.sections[next_idx].status = "in_progress"
            return self.document_outline.sections[next_idx]

        return None

    def revise_outline(self, completed_sections: List[str],
                      current_progress: str) -> DocumentOutline:
        """
        Revise the document outline based on what's been written.

        This allows adaptive planning - adjusting future sections based on
        how the document has actually developed.

        Args:
            completed_sections: List of completed section summaries
            current_progress: Summary of current state

        Returns:
            Revised DocumentOutline
        """
        if not self.document_outline:
            return self.document_outline

        completed_text = "\n".join([f"- {s}" for s in completed_sections])
        remaining_sections = self.document_outline.sections[
            self.document_outline.current_section_index:
        ]
        remaining_text = "\n".join([f"- {s.title}: {s.purpose}" for s in remaining_sections])

        prompt = f"""Review and revise the planned outline for a document in progress.

ORIGINAL DOCUMENT GOAL: {self.document_outline.goal}

COMPLETED SECTIONS:
{completed_text}

CURRENT PROGRESS:
{current_progress}

REMAINING PLANNED SECTIONS:
{remaining_text}

Based on what's been written, revise the remaining sections. You may:
1. Keep sections that still make sense
2. Modify section purposes or key points
3. Add new sections if gaps are discovered
4. Remove sections that are no longer relevant
5. Reorder sections for better flow

Format your response as:

[Revised Section 1]
Title: <section title>
Purpose: <updated purpose>
Key Points:
- <point 1>
- <point 2>
- <point 3>

[Revised Section 2]
...

Provide all remaining sections in the revised plan."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.6
        )

        # Parse revised sections
        revised_sections = []
        section_num = 1

        while f'[Revised Section {section_num}]' in response:
            try:
                next_section = f'[Revised Section {section_num + 1}]'
                section_content = get_content_between_a_b(
                    f'[Revised Section {section_num}]',
                    next_section if next_section in response else '',
                    response
                )

                title = get_content_between_a_b('Title:', 'Purpose:', section_content).strip()
                purpose = get_content_between_a_b('Purpose:', 'Key Points:', section_content).strip()
                key_points_text = section_content.split('Key Points:')[-1].strip()

                key_points = [
                    line.strip('- ').strip()
                    for line in key_points_text.split('\n')
                    if line.strip().startswith('-')
                ]

                revised_sections.append(SectionPlan(
                    title=title,
                    purpose=purpose,
                    key_points=key_points,
                    estimated_paragraphs=len(key_points)
                ))

                section_num += 1
            except:
                break

        # Replace remaining sections with revised ones
        if revised_sections:
            # Keep completed sections
            completed = self.document_outline.sections[:self.document_outline.current_section_index]
            # Add revised remaining sections
            self.document_outline.sections = completed + revised_sections

        return self.document_outline

    def get_section_context(self, section_index: int) -> str:
        """
        Get contextual information about a section's place in the document.

        Args:
            section_index: Index of the section

        Returns:
            Formatted context string
        """
        if not self.document_outline or section_index >= len(self.document_outline.sections):
            return ""

        section = self.document_outline.sections[section_index]
        context_parts = [
            f"DOCUMENT GOAL: {self.document_outline.goal}",
            f"\nCURRENT SECTION ({section_index + 1}/{len(self.document_outline.sections)}): {section.title}",
            f"PURPOSE: {section.purpose}"
        ]

        # Add previous section context
        if section_index > 0:
            prev_section = self.document_outline.sections[section_index - 1]
            context_parts.append(f"\nPREVIOUS SECTION: {prev_section.title}")

        # Add next section preview
        if section_index < len(self.document_outline.sections) - 1:
            next_section = self.document_outline.sections[section_index + 1]
            context_parts.append(f"NEXT SECTION: {next_section.title}")

        return "\n".join(context_parts)

    def get_progress_summary(self) -> Dict:
        """Get a summary of document progress"""
        if not self.document_outline:
            return {'progress': 0, 'status': 'not_started'}

        total = len(self.document_outline.sections)
        completed = sum(1 for s in self.document_outline.sections if s.status == "completed")
        current_idx = self.document_outline.current_section_index

        return {
            'progress': (completed / total * 100) if total > 0 else 0,
            'completed_sections': completed,
            'total_sections': total,
            'current_section': self.document_outline.sections[current_idx].title if current_idx < total else "Complete",
            'status': 'in_progress' if current_idx < total else 'completed'
        }
