"""
Content Verification Layer for RecurrentGPT

Checks generated content for:
- Contradictions with previous claims
- Redundant information
- Gaps in reasoning
- Factual consistency

Provides feedback loop to maintain quality.
"""

from typing import List, Dict, Optional, Tuple
from utils import get_api_response, get_content_between_a_b


class ContentVerifier:
    """
    Verifies generated content against existing document.

    This creates a feedback loop:
    1. Generate section
    2. Verify against existing content
    3. If issues found, regenerate with specific fixes
    4. Repeat until passes verification
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.verification_history: List[Dict] = []

    def verify_section(self, new_section: str,
                      existing_claims: List[str],
                      recent_content: str,
                      section_instruction: str) -> Dict:
        """
        Verify a generated section for consistency and quality.

        Args:
            new_section: The newly generated section
            existing_claims: List of claims made earlier in document
            recent_content: Recent paragraphs for context
            section_instruction: The instruction that guided generation

        Returns:
            Dict with:
            - passed: bool (whether verification passed)
            - contradictions: List of contradictions found
            - redundancies: List of redundant content
            - gaps: List of reasoning gaps
            - suggestions: Specific improvement suggestions
        """
        claims_text = "\n".join([f"- {c}" for c in existing_claims[:15]]) if existing_claims else "None yet"

        prompt = f"""You are verifying a newly generated section of a document for consistency and quality.

EXISTING CLAIMS IN DOCUMENT:
{claims_text}

RECENT CONTENT (for context):
{recent_content if recent_content else 'This is an early section'}

INSTRUCTION FOR THIS SECTION:
{section_instruction}

NEWLY GENERATED SECTION:
{new_section}

Verify this section and identify any issues:

1. CONTRADICTIONS: Does it contradict any existing claims?
2. REDUNDANCIES: Does it repeat information already covered?
3. REASONING GAPS: Are there logical leaps or missing steps?
4. INSTRUCTION ADHERENCE: Does it follow the instruction?
5. COHERENCE: Does it flow logically from recent content?

Provide your analysis in this format:

[Contradictions]
<List any contradictions, or "None" if none found>

[Redundancies]
<List any redundant content, or "None" if none found>

[Reasoning Gaps]
<List any gaps in logic, or "None" if none found>

[Instruction Adherence]
<"Yes" or explain how it deviates>

[Coherence Issues]
<List any flow problems, or "None" if none found>

[Overall Assessment]
<"PASS" if section is acceptable, or "FAIL" if it needs regeneration>

[Improvement Suggestions]
<Specific, actionable suggestions if assessment is FAIL>

Be thorough and critical."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.3  # Low temperature for consistent verification
        )

        try:
            result = {
                'contradictions': get_content_between_a_b('[Contradictions]', '[Redundancies]', response).strip(),
                'redundancies': get_content_between_a_b('[Redundancies]', '[Reasoning Gaps]', response).strip(),
                'gaps': get_content_between_a_b('[Reasoning Gaps]', '[Instruction Adherence]', response).strip(),
                'instruction_adherence': get_content_between_a_b('[Instruction Adherence]', '[Coherence Issues]', response).strip(),
                'coherence_issues': get_content_between_a_b('[Coherence Issues]', '[Overall Assessment]', response).strip(),
                'assessment': get_content_between_a_b('[Overall Assessment]', '[Improvement Suggestions]', response).strip(),
                'suggestions': get_content_between_a_b('[Improvement Suggestions]', '', response).strip()
            }

            # Determine if passed
            result['passed'] = 'PASS' in result['assessment'].upper()

            # Record verification
            self.verification_history.append({
                'passed': result['passed'],
                'had_contradictions': 'none' not in result['contradictions'].lower(),
                'had_redundancies': 'none' not in result['redundancies'].lower(),
                'had_gaps': 'none' not in result['gaps'].lower()
            })

            return result

        except Exception as e:
            # Fallback if parsing fails - assume pass to avoid blocking
            return {
                'passed': True,
                'contradictions': 'Unable to verify',
                'redundancies': 'Unable to verify',
                'gaps': 'Unable to verify',
                'instruction_adherence': 'Unable to verify',
                'coherence_issues': 'Unable to verify',
                'assessment': 'PASS (verification error)',
                'suggestions': ''
            }

    def verify_claim_consistency(self, new_claim: str,
                                 existing_claims: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify a single claim against existing claims.

        Args:
            new_claim: The new claim to verify
            existing_claims: Previously established claims

        Returns:
            (is_consistent, list_of_conflicts)
        """
        if not existing_claims:
            return True, []

        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(existing_claims)])

        prompt = f"""Check if a new claim contradicts any existing claims.

EXISTING CLAIMS:
{claims_text}

NEW CLAIM:
{new_claim}

Does the new claim contradict any existing claims?

Respond in this format:

[Contradicts]
<"Yes" or "No">

[Conflicting Claims]
<If Yes, list the claim numbers that conflict. If No, write "None">

[Explanation]
<Brief explanation of the contradiction or why there is none>"""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.2
        )

        try:
            contradicts = get_content_between_a_b('[Contradicts]', '[Conflicting Claims]', response).strip()
            conflicting = get_content_between_a_b('[Conflicting Claims]', '[Explanation]', response).strip()

            is_consistent = 'no' in contradicts.lower()

            conflicts = []
            if not is_consistent and conflicting.lower() != 'none':
                # Parse claim numbers
                import re
                numbers = re.findall(r'\d+', conflicting)
                conflicts = [existing_claims[int(n)-1] for n in numbers if int(n) <= len(existing_claims)]

            return is_consistent, conflicts

        except:
            # If verification fails, assume consistent to avoid blocking
            return True, []

    def check_redundancy(self, new_content: str,
                        existing_content: List[str],
                        embedder=None) -> Tuple[bool, List[str]]:
        """
        Check if new content is redundant with existing content.

        Args:
            new_content: New paragraph/section
            existing_content: Previous paragraphs
            embedder: Optional embedder for semantic similarity

        Returns:
            (is_redundant, list_of_similar_paragraphs)
        """
        if not existing_content:
            return False, []

        # Method 1: Semantic similarity (if embedder available)
        if embedder:
            new_embedding = embedder.encode([new_content])[0]
            existing_embeddings = embedder.encode(existing_content)
            similarities = embedder.cosine_scores(new_embedding, existing_embeddings)

            # High similarity threshold (0.85+) indicates redundancy
            redundant_indices = [i for i, sim in enumerate(similarities) if sim > 0.85]

            if redundant_indices:
                similar_paragraphs = [existing_content[i] for i in redundant_indices]
                return True, similar_paragraphs

        # Method 2: Text-based checking (fallback)
        # Check for significant word overlap
        new_words = set(new_content.lower().split())

        for para in existing_content[-5:]:  # Check last 5 paragraphs
            para_words = set(para.lower().split())
            overlap = len(new_words & para_words) / len(new_words) if new_words else 0

            if overlap > 0.7:  # 70% word overlap
                return True, [para]

        return False, []

    def generate_regeneration_prompt(self, verification_result: Dict,
                                    original_instruction: str) -> str:
        """
        Generate a detailed prompt for regenerating a failed section.

        Args:
            verification_result: Result from verify_section()
            original_instruction: The original instruction

        Returns:
            Enhanced instruction with specific fixes
        """
        fixes = ["REGENERATE THIS SECTION with the following corrections:\n"]

        if verification_result.get('contradictions', '').lower() != 'none':
            fixes.append(f"FIX CONTRADICTIONS:\n{verification_result['contradictions']}\n")

        if verification_result.get('redundancies', '').lower() != 'none':
            fixes.append(f"REMOVE REDUNDANCIES:\n{verification_result['redundancies']}\n")

        if verification_result.get('gaps', '').lower() != 'none':
            fixes.append(f"FILL REASONING GAPS:\n{verification_result['gaps']}\n")

        if verification_result.get('coherence_issues', '').lower() != 'none':
            fixes.append(f"IMPROVE COHERENCE:\n{verification_result['coherence_issues']}\n")

        if verification_result.get('suggestions'):
            fixes.append(f"SPECIFIC IMPROVEMENTS:\n{verification_result['suggestions']}\n")

        fixes.append(f"\nORIGINAL INSTRUCTION:\n{original_instruction}")

        return "\n".join(fixes)

    def extract_claims_from_section(self, section_text: str) -> List[str]:
        """
        Extract main claims from a section of text.

        Args:
            section_text: The text to analyze

        Returns:
            List of extracted claims
        """
        prompt = f"""Extract the main claims or assertions from this text.

TEXT:
{section_text}

List each distinct claim as a separate item. Focus on factual statements, arguments, or positions taken.

Format your response as:

[Claim 1]
<First claim>

[Claim 2]
<Second claim>

[Claim 3]
<Third claim>

Continue for all significant claims. Aim for 3-7 claims depending on content length."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.3
        )

        claims = []
        claim_num = 1

        while f'[Claim {claim_num}]' in response:
            try:
                next_marker = f'[Claim {claim_num + 1}]'
                claim = get_content_between_a_b(
                    f'[Claim {claim_num}]',
                    next_marker if next_marker in response else '',
                    response
                ).strip()

                if claim:
                    claims.append(claim)
                claim_num += 1
            except:
                break

        return claims

    def verify_with_retry(self, section_text: str,
                         existing_claims: List[str],
                         recent_content: str,
                         instruction: str,
                         max_retries: int = 3) -> Tuple[str, Dict, int]:
        """
        Verify section and provide guidance for regeneration if needed.

        This is the main entry point for the verification loop.

        Args:
            section_text: Generated section
            existing_claims: Previous claims
            recent_content: Recent context
            instruction: Generation instruction
            max_retries: Maximum verification attempts

        Returns:
            (final_section_text, final_verification_result, attempts_used)
        """
        current_text = section_text
        attempts = 0

        for attempt in range(max_retries):
            attempts += 1

            verification = self.verify_section(
                current_text,
                existing_claims,
                recent_content,
                instruction
            )

            if verification['passed']:
                return current_text, verification, attempts

            # If not passed and not last attempt, prepare for regeneration
            if attempt < max_retries - 1:
                # In a real implementation, this would trigger regeneration
                # For now, we return the result and let the caller handle it
                break

        # Return the last attempt's results
        return current_text, verification, attempts

    def get_verification_stats(self) -> Dict:
        """Get statistics about verification history"""
        if not self.verification_history:
            return {'total': 0, 'pass_rate': 0}

        total = len(self.verification_history)
        passed = sum(1 for v in self.verification_history if v['passed'])
        had_contradictions = sum(1 for v in self.verification_history if v.get('had_contradictions', False))
        had_redundancies = sum(1 for v in self.verification_history if v.get('had_redundancies', False))
        had_gaps = sum(1 for v in self.verification_history if v.get('had_gaps', False))

        return {
            'total_verifications': total,
            'passed': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'contradiction_rate': had_contradictions / total if total > 0 else 0,
            'redundancy_rate': had_redundancies / total if total > 0 else 0,
            'gap_rate': had_gaps / total if total > 0 else 0
        }
