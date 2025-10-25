"""
Enhanced RecurrentGPT Engine

Integrates all improvements:
1. Hierarchical memory system
2. Critique agent (replaces Human Simulator)
3. Multi-level planning
4. Dynamic generation parameters
5. Quality metrics and error handling
6. Content verification layer
7. Monitoring capabilities

This is the main orchestration layer that coordinates all components.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

from hierarchical_memory import HierarchicalMemory, Claim
from critique_agent import CritiqueAgent
from multi_level_planner import MultiLevelPlanner, DocumentOutline
from dynamic_parameters import DynamicParameterManager, GenerationParams
from quality_metrics import QualityMonitor, QualityMetrics
from content_verifier import ContentVerifier
from monitoring_dashboard import create_state_snapshot, CLIMonitor
from openai_embedder import OpenAIEmbedder
from utils import get_api_response, get_content_between_a_b


@dataclass
class GenerationResult:
    """Result of a generation step"""
    section_text: str
    success: bool
    attempts: int
    quality_score: float
    verification_passed: bool
    critique_score: int
    issues: List[str]
    metadata: Dict


class EnhancedRecurrentGPT:
    """
    Enhanced document generation system with comprehensive quality management.

    This system replaces the original RecurrentGPT's simple loop with:
    - Intelligent planning across multiple levels
    - Continuous quality monitoring
    - Automatic verification and regeneration
    - Adaptive parameter adjustment
    - Hierarchical memory management
    """

    def __init__(self, api_key: str, model: str = "gpt-4", writer_model: str = None):
        self.api_key = api_key
        self.model = model
        self.writer_model = writer_model or model

        # Initialize components
        self.embedder = OpenAIEmbedder(api_key)
        self.memory = HierarchicalMemory(embedder=self.embedder)
        self.planner = MultiLevelPlanner(api_key, model)
        self.critique_agent = CritiqueAgent(api_key, model)
        self.param_manager = DynamicParameterManager()
        self.quality_monitor = QualityMonitor(embedder=self.embedder)
        self.verifier = ContentVerifier(api_key, model)
        self.cli_monitor = CLIMonitor()

        # State
        self.current_section_id: Optional[str] = None
        self.generation_log: List[str] = []
        self.paused = False
        self.stop_requested = False

        # Configuration
        self.max_regeneration_attempts = 3
        self.auto_approve_threshold = 7  # Auto-approve sections with score >= 7
        self.enable_auto_approve = False

    def initialize_document(self, initial_input: str, num_sections: int = 10) -> DocumentOutline:
        """
        Initialize document generation.

        Args:
            initial_input: User's initial prompt/topic
            num_sections: Target number of sections

        Returns:
            Created DocumentOutline
        """
        self._log("Initializing document generation...")

        # Create document outline
        outline = self.planner.create_initial_outline(initial_input, num_sections)
        self._log(f"Created outline with {len(outline.sections)} sections")

        # Create root section in hierarchical memory
        root_section_id = self.memory.create_section(outline.title)
        self.current_section_id = root_section_id

        # Generate and analyze initial paragraphs
        initial_paragraphs = self._generate_initial_content(initial_input, outline)

        # Establish tone and scope
        if len(initial_paragraphs) >= 2:
            tone_scope = self.critique_agent.analyze_tone_and_scope(initial_paragraphs[:2])
            self._log(f"Established tone: {tone_scope['tone'][:100]}...")

        # Add initial paragraphs to memory
        for para in initial_paragraphs:
            self.memory.add_paragraph(root_section_id, para)

        return outline

    def _generate_initial_content(self, initial_input: str, outline: DocumentOutline) -> List[str]:
        """Generate initial 2-3 paragraphs to establish document"""
        prompt = f"""Begin writing a document on the following topic:

TOPIC: {initial_input}

DOCUMENT GOAL: {outline.goal}

FIRST SECTION: {outline.sections[0].title if outline.sections else 'Introduction'}

Write the first 2-3 paragraphs to introduce the topic and establish the document's tone and scope.

Format your response as:

[Paragraph 1]
<First paragraph>

[Paragraph 2]
<Second paragraph>

[Paragraph 3]
<Third paragraph - optional>

Each paragraph should be 3-5 sentences."""

        response = get_api_response(
            prompt,
            api_key=self.api_key,
            model=self.writer_model,
            temperature=0.75
        )

        paragraphs = []
        for i in range(1, 4):
            try:
                next_marker = f'[Paragraph {i+1}]' if i < 3 else ''
                para = get_content_between_a_b(
                    f'[Paragraph {i}]',
                    next_marker,
                    response
                ).strip()
                if para:
                    paragraphs.append(para)
            except:
                break

        return paragraphs if paragraphs else ["Failed to generate initial content."]

    def generate_next_section(self, section_index: int,
                             manual_instruction: Optional[str] = None) -> GenerationResult:
        """
        Generate the next section with full quality pipeline.

        Args:
            section_index: Index of section in outline
            manual_instruction: Optional manual override instruction

        Returns:
            GenerationResult with complete metadata
        """
        if self.stop_requested:
            return self._create_failed_result("Generation stopped by user")

        # Get current section plan
        if not self.planner.document_outline or section_index >= len(self.planner.document_outline.sections):
            return self._create_failed_result("No section to generate")

        section_plan = self.planner.document_outline.sections[section_index]
        self._log(f"\n{'='*60}")
        self._log(f"Generating section {section_index + 1}: {section_plan.title}")

        # Create section in memory
        section_id = self.memory.create_section(section_plan.title, self.current_section_id)
        self.current_section_id = section_id

        # Get generation parameters
        progress = self.planner.get_progress_summary()['progress']
        section_type = self._infer_section_type(section_plan.title, section_index)

        quality_trend_info = self.param_manager.get_quality_trend()
        params = self.param_manager.get_adaptive_params(
            progress_percent=progress,
            section_type=section_type,
            quality_trend=quality_trend_info[0],
            avg_quality=quality_trend_info[1]
        )

        self._log(f"Using temperature: {params.temperature}, detail: {params.instruction_detail_level}")

        # Plan next paragraphs
        completed_in_section = []
        paragraph_plans = self.planner.plan_next_paragraphs(section_index, completed_in_section, num_paragraphs=3)

        # Use manual instruction if provided, otherwise use first planned paragraph
        instruction = manual_instruction or paragraph_plans[0]

        # Generate with verification loop
        best_result = None
        for attempt in range(1, self.max_regeneration_attempts + 1):
            self._log(f"Generation attempt {attempt}/{self.max_regeneration_attempts}")

            # Adjust params for retry
            if attempt > 1 and best_result:
                params = self.param_manager.get_params_for_revision(
                    best_result.critique_score,
                    attempt
                )

            # Generate section
            section_text = self._generate_section_text(
                instruction=instruction,
                section_context=self.planner.get_section_context(section_index),
                params=params
            )

            if not section_text:
                continue

            # Verify content
            existing_claims = self._get_all_claims()
            recent_content = "\n\n".join(self.memory.paragraphs[-3:]) if self.memory.paragraphs else ""

            verification = self.verifier.verify_section(
                section_text,
                existing_claims,
                recent_content,
                instruction
            )

            # Critique
            critique = self.critique_agent.critique_section(
                section_text,
                instruction,
                recent_content,
                existing_claims
            )

            # Quality metrics
            quality_metrics = self.quality_monitor.evaluate_section(
                section_text,
                self.memory.paragraphs
            )

            # Create result
            result = GenerationResult(
                section_text=section_text,
                success=verification['passed'] and critique['overall_score'] >= 6,
                attempts=attempt,
                quality_score=quality_metrics[0].overall_score,
                verification_passed=verification['passed'],
                critique_score=critique['overall_score'],
                issues=quality_metrics[0].issues + ([] if verification['passed'] else ['Verification failed']),
                metadata={
                    'verification': verification,
                    'critique': critique,
                    'quality_metrics': quality_metrics[0].to_dict(),
                    'params': params.to_dict()
                }
            )

            # Check if acceptable
            if result.success:
                self._log(f"✓ Section approved (Quality: {result.quality_score:.1f}, Critique: {result.critique_score})")
                best_result = result
                break

            # Not acceptable - prepare for retry
            self._log(f"✗ Section needs improvement (Quality: {result.quality_score:.1f}, Critique: {result.critique_score})")

            if not best_result or result.critique_score > best_result.critique_score:
                best_result = result

            # Update instruction with feedback for next attempt
            if attempt < self.max_regeneration_attempts:
                instruction = self.verifier.generate_regeneration_prompt(verification, instruction)
                instruction += f"\n\nADDRESSING CRITIQUE:\n{critique['improvement_suggestions']}"

        # Use best result (even if not perfect)
        if best_result:
            # Add to memory
            claims_data = self._extract_claims_as_dicts(best_result.section_text)
            self.memory.add_paragraph(section_id, best_result.section_text, claims_data)

            # Update quality tracking
            self.param_manager.record_quality(best_result.quality_score)

            # Update monitoring
            self._update_monitor()

            return best_result

        return self._create_failed_result("Failed to generate acceptable section")

    def _generate_section_text(self, instruction: str,
                              section_context: str,
                              params: GenerationParams) -> Optional[str]:
        """Generate section text with given parameters"""
        # Get relevant context from hierarchical memory
        relevant_paragraphs = self.memory.get_relevant_paragraphs(
            instruction,
            top_k=3,
            section_id=self.current_section_id
        )

        related_content = "\n\n".join([f"From '{section}': {para[:200]}..."
                                      for para, score, section in relevant_paragraphs])

        # Get instruction template
        template = self.param_manager.get_instruction_template(params.instruction_detail_level)

        # Build prompt
        prompt = template.format(
            instruction=instruction,
            target_length="15-20",
            context=section_context,
            related_content=related_content if related_content else "None"
        )

        prompt += "\n\nWrite a coherent, well-structured section that addresses the instruction."

        # Generate
        try:
            response = get_api_response(
                prompt,
                api_key=self.api_key,
                model=self.writer_model,
                **params.to_dict()
            )
            return response.strip()
        except Exception as e:
            self._log(f"Generation error: {e}")
            return None

    def generate_full_document(self, initial_input: str,
                              num_sections: int = 10,
                              interactive: bool = False) -> str:
        """
        Generate a complete document.

        Args:
            initial_input: Starting prompt
            num_sections: Target number of sections
            interactive: If True, wait for approval between sections

        Returns:
            Complete document text
        """
        # Initialize
        outline = self.initialize_document(initial_input, num_sections)

        # Generate each section
        for section_idx in range(len(outline.sections)):
            if self.paused:
                self._log("Generation paused")
                while self.paused and not self.stop_requested:
                    import time
                    time.sleep(1)

            if self.stop_requested:
                self._log("Generation stopped")
                break

            result = self.generate_next_section(section_idx)

            if not result.success:
                self._log(f"Warning: Section {section_idx + 1} did not meet quality standards but proceeding")

            # Update planner
            self.planner.advance_to_next_section()

            # Interactive mode: wait for approval
            if interactive:
                approved = self._request_user_approval(result)
                if not approved:
                    # Regenerate
                    result = self.generate_next_section(section_idx)

            # Periodic plan revision (every 3 sections)
            if (section_idx + 1) % 3 == 0 and section_idx < len(outline.sections) - 1:
                self._maybe_revise_plan(section_idx)

        # Export document
        document_text = self.export_document()

        # Print summary
        self.cli_monitor.print_summary()

        return document_text

    def _get_all_claims(self) -> List[str]:
        """Get all claims from memory"""
        claims = []
        for section in self.memory.sections.values():
            for claim in section.claims:
                claims.append(claim.text)
        return claims

    def _extract_claims_as_dicts(self, text: str) -> List[Dict]:
        """Extract claims from text as dictionaries"""
        claim_texts = self.verifier.extract_claims_from_section(text)
        return [{'text': c, 'evidence': []} for c in claim_texts]

    def _infer_section_type(self, title: str, index: int) -> str:
        """Infer section type from title"""
        title_lower = title.lower()

        if 'intro' in title_lower or index == 0:
            return 'intro'
        elif 'conclu' in title_lower or 'summary' in title_lower:
            return 'conclusion'
        elif any(word in title_lower for word in ['technical', 'implementation', 'algorithm', 'method']):
            return 'technical'
        elif any(word in title_lower for word in ['story', 'example', 'case']):
            return 'narrative'
        else:
            return 'standard'

    def _maybe_revise_plan(self, current_section_idx: int):
        """Potentially revise the document plan based on progress"""
        if not self.planner.document_outline:
            return

        # Get summary of completed sections
        completed_summaries = []
        for i in range(current_section_idx + 1):
            if i < len(self.planner.document_outline.sections):
                section = self.planner.document_outline.sections[i]
                completed_summaries.append(f"{section.title}: {section.purpose}")

        # Get current progress
        progress_summary = f"Completed {current_section_idx + 1} sections. Quality trend: {self.param_manager.get_quality_trend()[0]}"

        # Revise
        self._log("Revising document plan based on progress...")
        self.planner.revise_outline(completed_summaries, progress_summary)

    def _request_user_approval(self, result: GenerationResult) -> bool:
        """Request user approval (in interactive mode)"""
        print(f"\n{'='*60}")
        print("SECTION GENERATED:")
        print(result.section_text[:300] + "..." if len(result.section_text) > 300 else result.section_text)
        print(f"\nQuality Score: {result.quality_score:.1f}/10")
        print(f"Critique Score: {result.critique_score}/10")

        if result.issues:
            print(f"Issues: {', '.join(result.issues)}")

        response = input("\nApprove? (y/n/e for edit): ").strip().lower()
        return response == 'y'

    def export_document(self, format: str = "markdown") -> str:
        """
        Export the complete document.

        Args:
            format: Export format (markdown, text, json, html)

        Returns:
            Formatted document string
        """
        if format == "markdown":
            return self._export_markdown()
        elif format == "text":
            return self._export_text()
        elif format == "json":
            import json
            return json.dumps(self.memory.export_to_dict(), indent=2)
        else:
            return self._export_text()

    def _export_markdown(self) -> str:
        """Export as markdown"""
        lines = []

        if self.planner.document_outline:
            lines.append(f"# {self.planner.document_outline.title}\n")

        for section in self.memory.sections.values():
            if section.depth == 0:
                lines.append(f"# {section.title}\n")
            else:
                lines.append(f"{'#' * (section.depth + 1)} {section.title}\n")

            if section.content:
                lines.append(section.content + "\n")

        return "\n".join(lines)

    def _export_text(self) -> str:
        """Export as plain text"""
        return "\n\n".join(self.memory.paragraphs)

    def _update_monitor(self):
        """Update monitoring dashboard/CLI"""
        snapshot = create_state_snapshot(
            hierarchical_memory=self.memory,
            planner=self.planner,
            quality_monitor=self.quality_monitor,
            verifier=self.verifier,
            param_manager=self.param_manager
        )
        self.cli_monitor.update(snapshot)

    def _create_failed_result(self, reason: str) -> GenerationResult:
        """Create a failed generation result"""
        return GenerationResult(
            section_text="",
            success=False,
            attempts=0,
            quality_score=0,
            verification_passed=False,
            critique_score=0,
            issues=[reason],
            metadata={}
        )

    def _log(self, message: str):
        """Log a message"""
        self.generation_log.append(message)
        print(message)

    # Control methods for dashboard
    def pause(self):
        """Pause generation"""
        self.paused = True

    def resume(self):
        """Resume generation"""
        self.paused = False

    def stop(self):
        """Stop generation"""
        self.stop_requested = True

    def get_state_snapshot(self) -> Dict:
        """Get current state for monitoring"""
        return create_state_snapshot(
            hierarchical_memory=self.memory,
            planner=self.planner,
            quality_monitor=self.quality_monitor,
            verifier=self.verifier,
            param_manager=self.param_manager
        )
