"""
Quality Metrics and Error Handling for RecurrentGPT

Implements:
- Semantic coherence tracking
- Vocabulary consistency analysis
- Structural balance metrics
- Proper retry logic with exponential backoff
- Quality thresholds and intervention triggers
"""

from typing import Dict, List, Optional, Tuple, Callable
import time
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class QualityMetrics:
    """Container for quality measurements"""
    semantic_coherence: float = 0.0  # 0-1 scale
    vocabulary_consistency: float = 0.0  # 0-1 scale
    structural_balance: float = 0.0  # 0-1 scale
    overall_score: float = 0.0  # 0-10 scale
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'semantic_coherence': self.semantic_coherence,
            'vocabulary_consistency': self.vocabulary_consistency,
            'structural_balance': self.structural_balance,
            'overall_score': self.overall_score,
            'issues': self.issues,
            'warnings': self.warnings
        }


class QualityAnalyzer:
    """
    Analyzes document quality across multiple dimensions.

    Provides quantitative metrics to detect degradation and trigger interventions.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.baseline_vocabulary: Optional[Counter] = None
        self.section_lengths: List[int] = []
        self.coherence_history: List[float] = []

    def analyze_section(self, section_text: str,
                       previous_sections: List[str] = None,
                       expected_vocabulary: Counter = None) -> QualityMetrics:
        """
        Comprehensive quality analysis of a section.

        Args:
            section_text: The section to analyze
            previous_sections: Previous sections for coherence checking
            expected_vocabulary: Expected vocabulary distribution

        Returns:
            QualityMetrics with scores and identified issues
        """
        metrics = QualityMetrics()

        # Analyze semantic coherence
        if previous_sections and self.embedder:
            metrics.semantic_coherence = self._analyze_semantic_coherence(
                section_text, previous_sections
            )
            self.coherence_history.append(metrics.semantic_coherence)
        else:
            metrics.semantic_coherence = 0.8  # Default for early sections

        # Analyze vocabulary consistency
        if expected_vocabulary or self.baseline_vocabulary:
            vocab_to_use = expected_vocabulary or self.baseline_vocabulary
            metrics.vocabulary_consistency = self._analyze_vocabulary_consistency(
                section_text, vocab_to_use
            )
        else:
            # Establish baseline from this section
            self.baseline_vocabulary = self._get_vocabulary_distribution(section_text)
            metrics.vocabulary_consistency = 1.0

        # Analyze structural balance
        metrics.structural_balance = self._analyze_structural_balance(section_text)

        # Calculate overall score (0-10 scale)
        metrics.overall_score = self._calculate_overall_score(metrics)

        # Identify issues
        metrics.issues = self._identify_issues(metrics, section_text)

        # Generate warnings
        metrics.warnings = self._generate_warnings(metrics)

        return metrics

    def _analyze_semantic_coherence(self, current_text: str,
                                    previous_sections: List[str]) -> float:
        """
        Measure semantic coherence with previous sections.

        Returns score 0-1 where 1 is perfectly coherent.
        """
        if not self.embedder or not previous_sections:
            return 0.8  # Default

        # Get embeddings
        current_embedding = self.embedder.encode([current_text])[0]

        # Compare with recent sections (last 3)
        recent_sections = previous_sections[-3:]
        recent_embeddings = self.embedder.encode(recent_sections)

        # Calculate average similarity
        similarities = self.embedder.cosine_scores(current_embedding, recent_embeddings)
        avg_similarity = sum(similarities) / len(similarities)

        # Convert to 0-1 scale (cosine similarity is already -1 to 1, but typically 0-1 for text)
        coherence = (avg_similarity + 1) / 2  # Normalize to 0-1

        return min(1.0, max(0.0, coherence))

    def _analyze_vocabulary_consistency(self, text: str,
                                       expected_vocab: Counter) -> float:
        """
        Measure vocabulary consistency with expected distribution.

        Returns score 0-1 where 1 is perfectly consistent.
        """
        current_vocab = self._get_vocabulary_distribution(text)

        # Calculate overlap
        common_words = set(current_vocab.keys()) & set(expected_vocab.keys())
        if not common_words:
            return 0.5  # Moderate score if no overlap

        # Calculate distribution similarity (using simplified KL divergence)
        total_current = sum(current_vocab.values())
        total_expected = sum(expected_vocab.values())

        similarity_scores = []
        for word in common_words:
            current_freq = current_vocab[word] / total_current
            expected_freq = expected_vocab[word] / total_expected
            # Ratio of frequencies (closer to 1 is better)
            ratio = min(current_freq, expected_freq) / max(current_freq, expected_freq)
            similarity_scores.append(ratio)

        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        # Penalize if too many unique words (vocabulary explosion)
        overlap_ratio = len(common_words) / len(current_vocab.keys())
        consistency_score = (avg_similarity * 0.7) + (overlap_ratio * 0.3)

        return min(1.0, max(0.0, consistency_score))

    def _get_vocabulary_distribution(self, text: str) -> Counter:
        """Extract vocabulary distribution (word frequencies)"""
        # Tokenize and normalize
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())  # Words 4+ chars
        return Counter(words)

    def _analyze_structural_balance(self, text: str) -> float:
        """
        Analyze structural balance (sentence length variation, paragraph structure).

        Returns score 0-1 where 1 is well-balanced.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return 0.7  # Not enough data, give moderate score

        # Calculate sentence length stats
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5

        # Good balance: avg 15-25 words, std_dev 5-12
        avg_score = 1.0 - abs(avg_length - 20) / 20  # Peaks at 20 words
        avg_score = max(0.0, min(1.0, avg_score))

        variance_score = 1.0 - abs(std_dev - 8) / 8  # Peaks at std_dev 8
        variance_score = max(0.0, min(1.0, variance_score))

        # Check for very short or very long sentences
        very_short = sum(1 for l in lengths if l < 5)
        very_long = sum(1 for l in lengths if l > 40)
        outlier_penalty = (very_short + very_long) / len(lengths)

        balance_score = (avg_score * 0.4 + variance_score * 0.4) * (1 - outlier_penalty * 0.5)

        # Record section length
        self.section_lengths.append(len(text.split()))

        # Check overall document balance
        if len(self.section_lengths) > 3:
            section_variance = self._calculate_variance(self.section_lengths[-5:])
            # Penalize if sections vary too much in length
            if section_variance > 10000:  # High variance in word count
                balance_score *= 0.9

        return max(0.0, min(1.0, balance_score))

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """
        Calculate overall quality score (0-10 scale).

        Weighted combination of all metrics.
        """
        # Weights
        coherence_weight = 0.4
        vocabulary_weight = 0.3
        structure_weight = 0.3

        # Weighted sum (0-1 scale)
        weighted_score = (
            metrics.semantic_coherence * coherence_weight +
            metrics.vocabulary_consistency * vocabulary_weight +
            metrics.structural_balance * structure_weight
        )

        # Convert to 0-10 scale
        return weighted_score * 10

    def _identify_issues(self, metrics: QualityMetrics, text: str) -> List[str]:
        """Identify specific quality issues"""
        issues = []

        if metrics.semantic_coherence < 0.6:
            issues.append("Low semantic coherence with previous sections")

        if metrics.vocabulary_consistency < 0.5:
            issues.append("Vocabulary inconsistency detected")

        if metrics.structural_balance < 0.5:
            issues.append("Poor structural balance")

        # Check for repetition
        if self._detect_repetition(text):
            issues.append("Excessive repetition detected")

        # Check for very short section
        word_count = len(text.split())
        if word_count < 50:
            issues.append(f"Section too short ({word_count} words)")

        # Check coherence trend
        if len(self.coherence_history) >= 3:
            recent_trend = self.coherence_history[-3:]
            if all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                issues.append("Coherence declining over recent sections")

        return issues

    def _detect_repetition(self, text: str) -> bool:
        """Detect excessive repetition in text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return False

        # Check for repeated sentences
        sentence_counts = Counter(sentences)
        if any(count > 1 for count in sentence_counts.values()):
            return True

        # Check for repeated phrases (5+ words)
        words = text.lower().split()
        phrases = [' '.join(words[i:i+5]) for i in range(len(words) - 4)]
        phrase_counts = Counter(phrases)

        # If any phrase appears 3+ times, flag as repetition
        if any(count >= 3 for count in phrase_counts.values()):
            return True

        return False

    def _generate_warnings(self, metrics: QualityMetrics) -> List[str]:
        """Generate warnings based on metrics"""
        warnings = []

        if metrics.overall_score < 5:
            warnings.append("CRITICAL: Overall quality below acceptable threshold")

        if metrics.semantic_coherence < 0.7:
            warnings.append("WARNING: Coherence with previous content is weak")

        if metrics.vocabulary_consistency < 0.6:
            warnings.append("WARNING: Vocabulary drift detected")

        if len(metrics.issues) > 3:
            warnings.append("ALERT: Multiple quality issues detected")

        return warnings


class RetryHandler:
    """
    Handles retries with exponential backoff and failure tracking.

    Replaces infinite retry loops with smart retry logic.
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.failure_counts: Dict[str, int] = {}

    def retry_with_backoff(self, func: Callable, operation_name: str,
                          *args, **kwargs) -> Optional[any]:
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            operation_name: Name for tracking failures
            *args, **kwargs: Arguments for the function

        Returns:
            Function result, or None if all retries failed
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)

                if result is not None:
                    # Success - reset failure count
                    if operation_name in self.failure_counts:
                        del self.failure_counts[operation_name]
                    return result

            except Exception as e:
                last_error = e

            # Failed - wait before retry
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)

        # All retries failed
        self.failure_counts[operation_name] = self.failure_counts.get(operation_name, 0) + 1
        return None

    def get_failure_count(self, operation_name: str) -> int:
        """Get number of consecutive failures for an operation"""
        return self.failure_counts.get(operation_name, 0)

    def reset_failures(self, operation_name: str):
        """Reset failure count for an operation"""
        if operation_name in self.failure_counts:
            del self.failure_counts[operation_name]


class QualityMonitor:
    """
    Monitors quality over time and triggers interventions.

    Combines quality analysis with threshold-based decision making.
    """

    def __init__(self, embedder=None):
        self.analyzer = QualityAnalyzer(embedder)
        self.retry_handler = RetryHandler()
        self.quality_history: List[QualityMetrics] = []
        self.intervention_log: List[Dict] = []

        # Thresholds
        self.min_acceptable_score = 6.0
        self.critical_score = 4.0
        self.max_consecutive_failures = 3

    def evaluate_section(self, section_text: str,
                        previous_sections: List[str] = None) -> Tuple[QualityMetrics, bool]:
        """
        Evaluate section quality and determine if intervention needed.

        Args:
            section_text: Section to evaluate
            previous_sections: Previous sections for context

        Returns:
            (QualityMetrics, should_regenerate)
        """
        metrics = self.analyzer.analyze_section(section_text, previous_sections)
        self.quality_history.append(metrics)

        should_regenerate = self._should_intervene(metrics)

        if should_regenerate:
            self.intervention_log.append({
                'section_number': len(self.quality_history),
                'score': metrics.overall_score,
                'issues': metrics.issues,
                'action': 'regenerate'
            })

        return metrics, should_regenerate

    def _should_intervene(self, metrics: QualityMetrics) -> bool:
        """Determine if intervention is needed"""
        # Critical quality
        if metrics.overall_score < self.critical_score:
            return True

        # Below acceptable threshold
        if metrics.overall_score < self.min_acceptable_score:
            return True

        # Multiple serious issues
        if len(metrics.issues) > 2:
            return True

        # Declining trend
        if len(self.quality_history) >= 3:
            recent_scores = [m.overall_score for m in self.quality_history[-3:]]
            if all(recent_scores[i] < recent_scores[i-1] - 0.5 for i in range(1, len(recent_scores))):
                return True

        return False

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        if not self.quality_history:
            return {'status': 'no_data'}

        recent_metrics = self.quality_history[-1]
        avg_score = sum(m.overall_score for m in self.quality_history) / len(self.quality_history)

        return {
            'current_score': recent_metrics.overall_score,
            'average_score': avg_score,
            'total_sections': len(self.quality_history),
            'interventions': len(self.intervention_log),
            'recent_issues': recent_metrics.issues,
            'coherence_trend': self.analyzer.coherence_history[-5:] if len(self.analyzer.coherence_history) >= 5 else self.analyzer.coherence_history,
            'status': 'healthy' if recent_metrics.overall_score >= self.min_acceptable_score else 'warning'
        }
