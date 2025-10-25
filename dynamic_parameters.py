"""
Dynamic Generation Parameters for RecurrentGPT

Adjusts generation parameters based on:
- Writing stage (beginning, middle, end)
- Section complexity
- Quality trends
- Document progress

This provides adaptive behavior instead of static parameters.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Parameters for text generation"""
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    instruction_detail_level: str = "medium"  # low, medium, high

    def to_dict(self) -> dict:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }


class DynamicParameterManager:
    """
    Manages dynamic adjustment of generation parameters.

    Parameters are adjusted based on:
    1. Document stage (early exploration vs late consistency)
    2. Section complexity (technical vs narrative)
    3. Quality trends (improving vs declining)
    4. Specific needs (revision vs new content)
    """

    def __init__(self):
        self.baseline_params = GenerationParams()
        self.current_params = GenerationParams()
        self.quality_history: list = []

    def get_params_for_stage(self, progress_percent: float,
                            section_type: str = "standard") -> GenerationParams:
        """
        Calculate generation parameters based on document progress.

        Args:
            progress_percent: How far through the document (0-100)
            section_type: Type of section (intro, body, conclusion, technical, narrative)

        Returns:
            GenerationParams optimized for this stage
        """
        params = GenerationParams()

        # Stage-based temperature adjustment
        if progress_percent < 20:
            # Early stage: Higher temperature for exploration
            params.temperature = 0.8
            params.instruction_detail_level = "high"  # More guidance early on
            params.top_p = 0.95

        elif progress_percent < 40:
            # Early-middle: Moderate temperature, establishing voice
            params.temperature = 0.75
            params.instruction_detail_level = "high"
            params.top_p = 0.92

        elif progress_percent < 70:
            # Middle: Lower temperature for consistency
            params.temperature = 0.65
            params.instruction_detail_level = "medium"
            params.top_p = 0.9
            params.frequency_penalty = 0.1  # Reduce repetition

        else:
            # Late stage: Low temperature for consistency and conclusion
            params.temperature = 0.55
            params.instruction_detail_level = "medium"
            params.top_p = 0.85
            params.frequency_penalty = 0.2  # Strong repetition penalty
            params.presence_penalty = 0.1  # Encourage novel wording

        # Section-type adjustments
        if section_type == "intro":
            params.temperature += 0.1  # More creativity for introductions
            params.instruction_detail_level = "high"

        elif section_type == "conclusion":
            params.temperature -= 0.1  # More consistency for conclusions
            params.frequency_penalty += 0.15  # Avoid repeating earlier content

        elif section_type == "technical":
            params.temperature -= 0.15  # More precision for technical content
            params.instruction_detail_level = "high"

        elif section_type == "narrative":
            params.temperature += 0.05  # Slightly more creativity for narrative

        # Clamp values to valid ranges
        params.temperature = max(0.1, min(1.0, params.temperature))
        params.top_p = max(0.5, min(1.0, params.top_p))
        params.frequency_penalty = max(0.0, min(2.0, params.frequency_penalty))
        params.presence_penalty = max(0.0, min(2.0, params.presence_penalty))

        self.current_params = params
        return params

    def get_params_for_revision(self, critique_score: int,
                               attempt_number: int = 1) -> GenerationParams:
        """
        Calculate parameters for regenerating a section.

        Args:
            critique_score: Score from critique agent (1-10)
            attempt_number: Which regeneration attempt this is

        Returns:
            GenerationParams optimized for revision
        """
        params = GenerationParams()

        # Base temperature on how bad the previous attempt was
        if critique_score <= 3:
            # Major issues: Higher temperature for different approach
            params.temperature = 0.8
            params.instruction_detail_level = "high"
        elif critique_score <= 6:
            # Moderate issues: Moderate temperature
            params.temperature = 0.7
            params.instruction_detail_level = "high"
        else:
            # Minor issues: Lower temperature for small adjustments
            params.temperature = 0.6
            params.instruction_detail_level = "medium"

        # Adjust for repeated attempts
        if attempt_number > 1:
            # Increase temperature with each attempt to explore different approaches
            params.temperature += (attempt_number - 1) * 0.1
            params.temperature = min(0.95, params.temperature)

        # Always use high detail for revisions
        params.instruction_detail_level = "high"
        params.frequency_penalty = 0.15  # Avoid repeating the failed attempt

        self.current_params = params
        return params

    def get_params_for_quality_trend(self, trend: str, avg_score: float) -> GenerationParams:
        """
        Adjust parameters based on quality trends.

        Args:
            trend: 'improving', 'declining', or 'stable'
            avg_score: Average quality score (1-10)

        Returns:
            GenerationParams adjusted for quality trend
        """
        params = self.current_params

        if trend == "declining":
            # Quality is declining: increase guidance
            params.instruction_detail_level = "high"
            params.temperature -= 0.1  # More conservative
            params.frequency_penalty += 0.1  # Reduce repetition

        elif trend == "improving":
            # Quality is improving: maintain or slightly relax
            if avg_score >= 8:
                params.instruction_detail_level = "medium"
                params.temperature += 0.05  # Slightly more freedom

        # Always clamp to valid ranges
        params.temperature = max(0.1, min(1.0, params.temperature))
        params.frequency_penalty = max(0.0, min(2.0, params.frequency_penalty))

        self.current_params = params
        return params

    def get_instruction_template(self, detail_level: str) -> str:
        """
        Get instruction template based on detail level.

        Args:
            detail_level: 'low', 'medium', or 'high'

        Returns:
            Template string with appropriate guidance level
        """
        if detail_level == "high":
            return """DETAILED INSTRUCTIONS:

{instruction}

REQUIREMENTS:
- Follow the established tone and style
- Ensure logical flow from previous paragraphs
- Address all key points mentioned
- Maintain consistency with previous claims
- Use specific examples and evidence where appropriate
- Aim for {target_length} sentences

CONTEXT FOR THIS SECTION:
{context}

RELATED PREVIOUS CONTENT:
{related_content}"""

        elif detail_level == "medium":
            return """INSTRUCTIONS:
{instruction}

REQUIREMENTS:
- Maintain established tone and style
- Ensure logical flow
- Aim for {target_length} sentences

CONTEXT:
{context}"""

        else:  # low
            return """INSTRUCTIONS:
{instruction}

Context: {context}"""

    def adjust_for_complexity(self, section_complexity: str) -> GenerationParams:
        """
        Adjust parameters based on estimated section complexity.

        Args:
            section_complexity: 'simple', 'moderate', 'complex'

        Returns:
            Adjusted GenerationParams
        """
        params = self.current_params

        if section_complexity == "complex":
            params.temperature -= 0.1  # More precision
            params.max_tokens += 500  # May need more space
            params.instruction_detail_level = "high"

        elif section_complexity == "simple":
            params.temperature += 0.05  # Slight creativity boost
            params.max_tokens -= 200  # Likely shorter
            params.instruction_detail_level = "medium"

        # Clamp values
        params.temperature = max(0.1, min(1.0, params.temperature))
        params.max_tokens = max(500, min(4000, params.max_tokens))

        self.current_params = params
        return params

    def get_adaptive_params(self,
                           progress_percent: float,
                           section_type: str = "standard",
                           quality_trend: Optional[str] = None,
                           avg_quality: Optional[float] = None,
                           is_revision: bool = False,
                           revision_attempt: int = 1,
                           critique_score: Optional[int] = None) -> GenerationParams:
        """
        Get generation parameters considering all factors.

        This is the main method that combines all adjustment strategies.

        Args:
            progress_percent: Document progress (0-100)
            section_type: Type of section
            quality_trend: Quality trend if available
            avg_quality: Average quality score if available
            is_revision: Whether this is a revision
            revision_attempt: Which revision attempt
            critique_score: Score if this is a revision

        Returns:
            Optimally adjusted GenerationParams
        """
        if is_revision and critique_score is not None:
            params = self.get_params_for_revision(critique_score, revision_attempt)
        else:
            params = self.get_params_for_stage(progress_percent, section_type)

        # Apply quality trend adjustments if available
        if quality_trend and avg_quality:
            params = self.get_params_for_quality_trend(quality_trend, avg_quality)

        return params

    def record_quality(self, score: float):
        """Record a quality score for trend analysis"""
        self.quality_history.append(score)

    def get_quality_trend(self) -> tuple:
        """
        Get current quality trend.

        Returns:
            (trend_direction, average_score)
            trend_direction: 'improving', 'declining', 'stable', or 'insufficient_data'
        """
        if len(self.quality_history) < 3:
            return ('insufficient_data', 0.0)

        recent = self.quality_history[-3:]
        older = self.quality_history[:-3] if len(self.quality_history) > 3 else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        overall_avg = sum(self.quality_history) / len(self.quality_history)

        if recent_avg > older_avg + 0.5:
            return ('improving', overall_avg)
        elif recent_avg < older_avg - 0.5:
            return ('declining', overall_avg)
        else:
            return ('stable', overall_avg)

    def reset_to_baseline(self):
        """Reset parameters to baseline"""
        self.current_params = GenerationParams()

    def get_current_summary(self) -> dict:
        """Get summary of current parameter state"""
        trend, avg = self.get_quality_trend()

        return {
            'temperature': self.current_params.temperature,
            'instruction_detail': self.current_params.instruction_detail_level,
            'quality_trend': trend,
            'avg_quality': avg,
            'total_generations': len(self.quality_history)
        }
