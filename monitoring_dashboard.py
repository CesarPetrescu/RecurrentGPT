"""
Monitoring Dashboard for RecurrentGPT

Provides real-time visibility and control:
- Current state of all memory components
- Quality metrics visualization
- Manual intervention controls
- Parameter adjustment
- Section approval workflow
"""

import gradio as gr
from typing import Dict, List, Optional, Callable
import json


class MonitoringDashboard:
    """
    Interactive dashboard for monitoring and controlling document generation.

    Allows users to:
    - View current document state
    - See quality metrics in real-time
    - Intervene to adjust parameters
    - Approve/reject sections
    - Provide guidance
    """

    def __init__(self):
        self.state_data: Dict = {}
        self.callbacks: Dict[str, Callable] = {}
        self.intervention_enabled = True

    def register_callback(self, event_name: str, callback: Callable):
        """Register a callback for dashboard events"""
        self.callbacks[event_name] = callback

    def update_state(self, state_data: Dict):
        """Update dashboard state data"""
        self.state_data = state_data

    def create_dashboard(self) -> gr.Blocks:
        """
        Create the Gradio dashboard interface.

        Returns:
            gr.Blocks dashboard
        """
        with gr.Blocks(title="RecurrentGPT Monitor", theme=gr.themes.Soft()) as dashboard:
            gr.Markdown("# RecurrentGPT Monitoring Dashboard")
            gr.Markdown("Real-time monitoring and control for document generation")

            with gr.Tabs():
                # Tab 1: Overview
                with gr.Tab("Overview"):
                    with gr.Row():
                        with gr.Column():
                            progress_bar = gr.Slider(
                                minimum=0, maximum=100, value=0,
                                label="Document Progress (%)",
                                interactive=False
                            )
                            status_text = gr.Textbox(
                                label="Current Status",
                                value="Initializing...",
                                interactive=False
                            )

                        with gr.Column():
                            quality_score = gr.Number(
                                label="Current Quality Score (0-10)",
                                value=0,
                                interactive=False
                            )
                            avg_quality = gr.Number(
                                label="Average Quality Score",
                                value=0,
                                interactive=False
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Document Outline")
                            outline_text = gr.Textbox(
                                label="Current Outline",
                                lines=15,
                                interactive=False
                            )

                        with gr.Column():
                            gr.Markdown("### Quality Metrics")
                            coherence_score = gr.Number(label="Semantic Coherence", value=0, interactive=False)
                            vocabulary_score = gr.Number(label="Vocabulary Consistency", value=0, interactive=False)
                            structure_score = gr.Number(label="Structural Balance", value=0, interactive=False)

                    refresh_btn = gr.Button("🔄 Refresh Dashboard")

                # Tab 2: Current Section
                with gr.Tab("Current Section"):
                    with gr.Row():
                        with gr.Column():
                            section_title = gr.Textbox(label="Section Title", interactive=False)
                            section_instruction = gr.Textbox(
                                label="Generation Instruction",
                                lines=3,
                                interactive=False
                            )

                        with gr.Column():
                            section_status = gr.Textbox(label="Status", interactive=False)
                            generation_attempt = gr.Number(label="Generation Attempt", value=1, interactive=False)

                    section_content = gr.Textbox(
                        label="Generated Content",
                        lines=12,
                        interactive=False
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Verification Results")
                            verification_status = gr.Textbox(label="Verification", interactive=False)
                            contradictions = gr.Textbox(label="Contradictions", lines=2, interactive=False)
                            redundancies = gr.Textbox(label="Redundancies", lines=2, interactive=False)

                        with gr.Column():
                            gr.Markdown("### Critique")
                            critique_score = gr.Number(label="Critique Score", value=0, interactive=False)
                            critique_weaknesses = gr.Textbox(label="Weaknesses", lines=3, interactive=False)
                            critique_suggestions = gr.Textbox(label="Suggestions", lines=3, interactive=False)

                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve Section", variant="primary")
                        reject_btn = gr.Button("❌ Reject & Regenerate", variant="stop")
                        edit_btn = gr.Button("✏️ Edit & Approve")

                    action_result = gr.Textbox(label="Action Result", interactive=False)

                # Tab 3: Memory State
                with gr.Tab("Memory State"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Short-Term Memory")
                            short_memory = gr.Textbox(
                                label="Current Summary",
                                lines=8,
                                interactive=True,
                                placeholder="Short-term memory summary..."
                            )
                            update_short_memory_btn = gr.Button("Update Short Memory")

                        with gr.Column():
                            gr.Markdown("### Hierarchical Structure")
                            memory_stats = gr.JSON(label="Memory Statistics")

                    gr.Markdown("### Long-Term Memory (Recent Paragraphs)")
                    long_memory = gr.Textbox(
                        label="Recent Paragraphs",
                        lines=10,
                        interactive=False
                    )

                    gr.Markdown("### Claims Database")
                    claims_list = gr.JSON(label="Established Claims")

                # Tab 4: Planning
                with gr.Tab("Planning"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Document Plan")
                            doc_goal = gr.Textbox(label="Document Goal", lines=2, interactive=False)
                            remaining_sections = gr.JSON(label="Remaining Sections")

                        with gr.Column():
                            gr.Markdown("### Next Paragraphs")
                            next_plans = gr.JSON(label="Upcoming Paragraph Plans")

                    with gr.Row():
                        revise_plan_btn = gr.Button("🔄 Revise Remaining Plan")
                        add_section_btn = gr.Button("➕ Add Section")

                    custom_instruction = gr.Textbox(
                        label="Custom Instruction for Next Section",
                        lines=3,
                        placeholder="Provide specific instruction to override planned next step..."
                    )
                    apply_instruction_btn = gr.Button("Apply Custom Instruction")

                # Tab 5: Parameters
                with gr.Tab("Parameters"):
                    gr.Markdown("### Generation Parameters")

                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.05,
                                label="Temperature"
                            )
                            top_p = gr.Slider(
                                minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                                label="Top P"
                            )

                        with gr.Column():
                            freq_penalty = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                                label="Frequency Penalty"
                            )
                            presence_penalty = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                                label="Presence Penalty"
                            )

                    detail_level = gr.Radio(
                        choices=["low", "medium", "high"],
                        value="medium",
                        label="Instruction Detail Level"
                    )

                    with gr.Row():
                        apply_params_btn = gr.Button("Apply Parameters")
                        reset_params_btn = gr.Button("Reset to Adaptive")

                    param_result = gr.Textbox(label="Parameter Update Result", interactive=False)

                # Tab 6: Logs & History
                with gr.Tab("History"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Quality History")
                            quality_history_plot = gr.LinePlot(
                                x="section",
                                y="score",
                                title="Quality Scores Over Time",
                                width=400,
                                height=300
                            )

                        with gr.Column():
                            gr.Markdown("### Interventions")
                            intervention_log = gr.JSON(label="Intervention Log")

                    gr.Markdown("### Generation Log")
                    generation_log = gr.Textbox(
                        label="Recent Events",
                        lines=12,
                        interactive=False
                    )

                # Tab 7: Control
                with gr.Tab("Control"):
                    gr.Markdown("### Generation Control")

                    with gr.Row():
                        pause_btn = gr.Button("⏸️ Pause Generation", variant="secondary")
                        resume_btn = gr.Button("▶️ Resume Generation", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop & Export", variant="stop")

                    auto_approve = gr.Checkbox(
                        label="Auto-approve sections above quality threshold",
                        value=False
                    )
                    quality_threshold = gr.Slider(
                        minimum=1, maximum=10, value=6, step=1,
                        label="Auto-approve threshold"
                    )

                    gr.Markdown("### Export")
                    export_format = gr.Radio(
                        choices=["markdown", "text", "json", "html"],
                        value="markdown",
                        label="Export Format"
                    )
                    export_btn = gr.Button("📥 Export Document")
                    export_output = gr.File(label="Download")

            # Event handlers
            def handle_refresh():
                """Refresh dashboard with current state"""
                return self._get_current_state()

            def handle_approve():
                """Approve current section"""
                if 'on_approve' in self.callbacks:
                    result = self.callbacks['on_approve']()
                    return f"Section approved: {result}"
                return "No callback registered"

            def handle_reject():
                """Reject and regenerate current section"""
                if 'on_reject' in self.callbacks:
                    result = self.callbacks['on_reject']()
                    return f"Section rejected, regenerating: {result}"
                return "No callback registered"

            def handle_apply_params(temp, top_p_val, freq_pen, pres_pen, detail):
                """Apply custom parameters"""
                params = {
                    'temperature': temp,
                    'top_p': top_p_val,
                    'frequency_penalty': freq_pen,
                    'presence_penalty': pres_pen,
                    'instruction_detail_level': detail
                }
                if 'on_params_change' in self.callbacks:
                    result = self.callbacks['on_params_change'](params)
                    return f"Parameters updated: {result}"
                return f"Parameters set to: {params}"

            # Connect event handlers
            refresh_btn.click(
                fn=handle_refresh,
                outputs=[
                    progress_bar, status_text, quality_score, avg_quality,
                    outline_text, coherence_score, vocabulary_score, structure_score
                ]
            )

            approve_btn.click(fn=handle_approve, outputs=action_result)
            reject_btn.click(fn=handle_reject, outputs=action_result)

            apply_params_btn.click(
                fn=handle_apply_params,
                inputs=[temperature, top_p, freq_penalty, presence_penalty, detail_level],
                outputs=param_result
            )

        return dashboard

    def _get_current_state(self) -> tuple:
        """Get current state for dashboard display"""
        # Extract values from state_data
        progress = self.state_data.get('progress', 0)
        status = self.state_data.get('status', 'Unknown')
        current_quality = self.state_data.get('current_quality_score', 0)
        avg_quality = self.state_data.get('average_quality_score', 0)
        outline = self.state_data.get('outline', 'No outline available')
        coherence = self.state_data.get('coherence_score', 0)
        vocabulary = self.state_data.get('vocabulary_score', 0)
        structure = self.state_data.get('structure_score', 0)

        return (
            progress, status, current_quality, avg_quality,
            outline, coherence, vocabulary, structure
        )

    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the dashboard"""
        dashboard = self.create_dashboard()
        dashboard.launch(share=share, server_port=server_port)


class CLIMonitor:
    """
    Simplified command-line monitor for non-interactive use.

    Provides periodic status updates without GUI.
    """

    def __init__(self):
        self.last_status: Dict = {}

    def update(self, state_data: Dict):
        """Print status update to CLI"""
        self.last_status = state_data

        print("\n" + "="*60)
        print(f"STATUS: {state_data.get('status', 'Unknown')}")
        print(f"Progress: {state_data.get('progress', 0):.1f}%")
        print(f"Current Section: {state_data.get('current_section', 'N/A')}")
        print(f"Quality Score: {state_data.get('current_quality_score', 0):.2f}/10")

        if state_data.get('issues'):
            print(f"\nIssues: {', '.join(state_data['issues'])}")

        if state_data.get('warnings'):
            print(f"Warnings: {', '.join(state_data['warnings'])}")

        print("="*60 + "\n")

    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print(f"Total Sections: {self.last_status.get('total_sections', 0)}")
        print(f"Average Quality: {self.last_status.get('average_quality_score', 0):.2f}/10")
        print(f"Interventions: {self.last_status.get('interventions', 0)}")
        print("="*60 + "\n")


def create_state_snapshot(hierarchical_memory=None,
                         planner=None,
                         quality_monitor=None,
                         verifier=None,
                         param_manager=None) -> Dict:
    """
    Create a state snapshot from all system components.

    Args:
        hierarchical_memory: HierarchicalMemory instance
        planner: MultiLevelPlanner instance
        quality_monitor: QualityMonitor instance
        verifier: ContentVerifier instance
        param_manager: DynamicParameterManager instance

    Returns:
        Dict with complete state snapshot
    """
    snapshot = {
        'status': 'active',
        'progress': 0,
        'current_section': 'Unknown',
        'current_quality_score': 0,
        'average_quality_score': 0,
        'outline': '',
        'coherence_score': 0,
        'vocabulary_score': 0,
        'structure_score': 0,
        'issues': [],
        'warnings': []
    }

    # Add planner data
    if planner and planner.document_outline:
        progress_data = planner.get_progress_summary()
        snapshot['progress'] = progress_data['progress']
        snapshot['current_section'] = progress_data['current_section']
        snapshot['status'] = progress_data['status']
        snapshot['outline'] = planner.document_outline.get_outline_tree() if hasattr(planner.document_outline, 'get_outline_tree') else ''

    # Add quality data
    if quality_monitor:
        dashboard_data = quality_monitor.get_dashboard_data()
        snapshot['current_quality_score'] = dashboard_data.get('current_score', 0)
        snapshot['average_quality_score'] = dashboard_data.get('average_score', 0)
        snapshot['issues'] = dashboard_data.get('recent_issues', [])

        if quality_monitor.quality_history:
            latest = quality_monitor.quality_history[-1]
            snapshot['coherence_score'] = latest.semantic_coherence
            snapshot['vocabulary_score'] = latest.vocabulary_consistency
            snapshot['structure_score'] = latest.structural_balance
            snapshot['warnings'] = latest.warnings

    # Add parameter data
    if param_manager:
        param_summary = param_manager.get_current_summary()
        snapshot['temperature'] = param_summary['temperature']
        snapshot['instruction_detail'] = param_summary['instruction_detail']

    # Add memory data
    if hierarchical_memory:
        memory_summary = hierarchical_memory.get_document_summary()
        snapshot['total_sections'] = memory_summary['total_sections']
        snapshot['total_paragraphs'] = memory_summary['total_paragraphs']

    return snapshot
