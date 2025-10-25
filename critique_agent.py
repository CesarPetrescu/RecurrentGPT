"""
Critique Agent for RecurrentGPT

Replaces the Human Simulator's text expansion role with genuine critique and feedback.
Identifies weaknesses, inconsistencies, and areas for improvement.
"""

from typing import Dict, List, Optional, Tuple
from utils import get_api_response, get_content_between_a_b


class CritiqueAgent:
    """
    Provides critical analysis of generated content.

    Instead of expanding text, this agent:
    - Identifies logical weaknesses
    - Checks tone and scope consistency
    - Provides actionable feedback
    - Evaluates instruction quality
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.established_tone: Optional[str] = None
        self.established_scope: Optional[str] = None
        self.critique_history: List[Dict] = []

    def analyze_tone_and_scope(self, initial_paragraphs: List[str]) -> Dict[str, str]:
        """
        Analyze the initial paragraphs to establish baseline tone and scope.

        Args:
            initial_paragraphs: First 2-3 paragraphs of the document

        Returns:
            Dict with 'tone' and 'scope' descriptions
        """
        content = "\n\n".join(initial_paragraphs)

        prompt = f"""Analyze the following initial paragraphs of a document and describe:

1. TONE: The writing style (formal/informal, technical/accessible, objective/subjective, etc.)
2. SCOPE: The intended breadth and depth (overview/detailed, narrow/broad, introductory/advanced)

Initial Paragraphs:
{content}

Provide your analysis in this format:

[Tone Analysis]
<2-3 sentences describing the tone>

[Scope Analysis]
<2-3 sentences describing the scope>

Be specific and use this analysis to ensure consistency in future sections."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.3  # Low temperature for consistent analysis
        )

        try:
            tone = get_content_between_a_b('[Tone Analysis]', '[Scope Analysis]', response)
            scope = get_content_between_a_b('[Scope Analysis]', '', response).strip()

            self.established_tone = tone
            self.established_scope = scope

            return {'tone': tone, 'scope': scope}
        except:
            # Fallback if parsing fails
            self.established_tone = "Unable to establish baseline tone"
            self.established_scope = "Unable to establish baseline scope"
            return {'tone': self.established_tone, 'scope': self.established_scope}

    def critique_section(self, section_text: str, instruction: str,
                        recent_context: str = "", claims: List[str] = None) -> Dict:
        """
        Provide detailed critique of a generated section.

        Args:
            section_text: The generated section to critique
            instruction: The instruction that guided this section
            recent_context: Recent paragraphs for context
            claims: Previously established claims (for consistency checking)

        Returns:
            Dict containing:
            - weaknesses: List of identified issues
            - strengths: What works well
            - consistency_issues: Tone/scope problems
            - contradictions: Conflicts with established claims
            - improvement_suggestions: Specific actionable feedback
            - overall_score: 1-10 rating
        """
        claims_text = ""
        if claims:
            claims_text = "\n".join([f"- {c}" for c in claims[:10]])  # Limit to 10

        prompt = f"""You are a critical document reviewer. Analyze the following generated section for quality and consistency.

ESTABLISHED DOCUMENT CHARACTERISTICS:
Tone: {self.established_tone or 'Not yet established'}
Scope: {self.established_scope or 'Not yet established'}

PREVIOUS CLAIMS MADE IN DOCUMENT:
{claims_text if claims_text else 'None yet'}

RECENT CONTEXT (previous paragraphs):
{recent_context if recent_context else 'This is one of the first sections'}

INSTRUCTION FOR THIS SECTION:
{instruction}

GENERATED SECTION:
{section_text}

Provide a thorough critique using this format:

[Strengths]
<What works well in this section - be specific>

[Weaknesses]
<Logical gaps, unclear arguments, weak evidence, poor structure>

[Consistency Issues]
<Does it match the established tone and scope? If not, how does it deviate?>

[Contradictions]
<Any statements that contradict previously established claims?>

[Improvement Suggestions]
<Specific, actionable recommendations to address the weaknesses>

[Overall Score]
<Rate 1-10, where 10 is publication-ready and 1 is needs complete rewrite>

Be honest and critical. The goal is to improve the document, not validate poor work."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.4
        )

        try:
            critique = {
                'strengths': get_content_between_a_b('[Strengths]', '[Weaknesses]', response).strip(),
                'weaknesses': get_content_between_a_b('[Weaknesses]', '[Consistency Issues]', response).strip(),
                'consistency_issues': get_content_between_a_b('[Consistency Issues]', '[Contradictions]', response).strip(),
                'contradictions': get_content_between_a_b('[Contradictions]', '[Improvement Suggestions]', response).strip(),
                'improvement_suggestions': get_content_between_a_b('[Improvement Suggestions]', '[Overall Score]', response).strip(),
                'overall_score': self._extract_score(get_content_between_a_b('[Overall Score]', '', response))
            }

            self.critique_history.append({
                'section': section_text[:100] + '...',
                'score': critique['overall_score']
            })

            return critique

        except Exception as e:
            # Return minimal critique if parsing fails
            return {
                'strengths': 'Unable to parse',
                'weaknesses': 'Parsing error',
                'consistency_issues': 'Unable to assess',
                'contradictions': 'Unable to assess',
                'improvement_suggestions': 'Regenerate section',
                'overall_score': 5
            }

    def _extract_score(self, score_text: str) -> int:
        """Extract numeric score from text"""
        import re
        match = re.search(r'(\d+)', score_text)
        if match:
            score = int(match.group(1))
            return max(1, min(10, score))  # Clamp to 1-10
        return 5  # Default to middle score

    def evaluate_instructions(self, instructions: List[str], current_outline: str,
                            recent_sections: List[str]) -> Dict:
        """
        Evaluate proposed instructions and select/improve the best one.

        Args:
            instructions: List of 3 proposed instructions
            current_outline: Current document outline
            recent_sections: Recent section titles for context

        Returns:
            Dict with:
            - selected_instruction: The best instruction (possibly revised)
            - reasoning: Why this instruction was chosen
            - revisions: Any improvements made
        """
        instructions_text = "\n".join([f"{i+1}. {inst}" for i, inst in enumerate(instructions)])
        recent_text = "\n".join([f"- {s}" for s in recent_sections[-5:]])

        prompt = f"""You are planning the next section of a document. Evaluate these three proposed instructions and select the best one, or revise it to be better.

DOCUMENT OUTLINE SO FAR:
{current_outline}

RECENT SECTIONS:
{recent_text}

PROPOSED INSTRUCTIONS:
{instructions_text}

Evaluate based on:
1. Logical flow - Does it follow naturally from previous sections?
2. Coherence - Does it advance the document's goals?
3. Avoiding redundancy - Does it cover new ground?
4. Appropriate scope - Is it neither too broad nor too narrow?

Provide your evaluation in this format:

[Selected Instruction Number]
<1, 2, or 3>

[Reasoning]
<Why this instruction is best, or why all are problematic>

[Revised Instruction]
<The selected instruction, improved if needed - or a completely new one if all three are poor>

Be willing to significantly revise or reject all options if they don't serve the document well."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.5
        )

        try:
            selected_num_text = get_content_between_a_b('[Selected Instruction Number]', '[Reasoning]', response).strip()
            reasoning = get_content_between_a_b('[Reasoning]', '[Revised Instruction]', response).strip()
            revised_instruction = get_content_between_a_b('[Revised Instruction]', '', response).strip()

            return {
                'selected_instruction': revised_instruction,
                'reasoning': reasoning,
                'original_number': selected_num_text
            }

        except Exception as e:
            # Fallback: just return first instruction
            return {
                'selected_instruction': instructions[0] if instructions else "Continue writing the document.",
                'reasoning': 'Parsing failed, using fallback',
                'original_number': '1'
            }

    def should_regenerate(self, critique: Dict, threshold: int = 6) -> bool:
        """
        Determine if a section should be regenerated based on critique.

        Args:
            critique: Output from critique_section()
            threshold: Minimum acceptable score (default 6/10)

        Returns:
            True if section should be regenerated
        """
        return critique['overall_score'] < threshold

    def get_regeneration_guidance(self, critique: Dict) -> str:
        """
        Convert critique into specific guidance for regeneration.

        Args:
            critique: Output from critique_section()

        Returns:
            Formatted guidance string for the Writer
        """
        guidance_parts = ["GUIDANCE FOR IMPROVEMENT:"]

        if critique.get('weaknesses'):
            guidance_parts.append(f"\nAddress these weaknesses:\n{critique['weaknesses']}")

        if critique.get('consistency_issues'):
            guidance_parts.append(f"\nConsistency requirements:\n{critique['consistency_issues']}")

        if critique.get('contradictions'):
            guidance_parts.append(f"\nResolve these contradictions:\n{critique['contradictions']}")

        if critique.get('improvement_suggestions'):
            guidance_parts.append(f"\nSpecific improvements:\n{critique['improvement_suggestions']}")

        return "\n".join(guidance_parts)

    def get_quality_trend(self) -> Dict:
        """
        Analyze the trend in section quality over time.

        Returns:
            Dict with trend analysis
        """
        if len(self.critique_history) < 2:
            return {'trend': 'insufficient_data', 'avg_score': 0}

        scores = [c['score'] for c in self.critique_history]
        recent_scores = scores[-3:]  # Last 3 sections
        older_scores = scores[:-3] if len(scores) > 3 else scores

        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores) if older_scores else recent_avg

        if recent_avg > older_avg + 0.5:
            trend = 'improving'
        elif recent_avg < older_avg - 0.5:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'avg_score': sum(scores) / len(scores),
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'total_sections': len(self.critique_history)
        }
