#!/usr/bin/env python3
"""
Enhanced RecurrentGPT Gradio Interface

Interactive web interface with full monitoring dashboard.

Usage:
    python enhanced_gradio_server.py
"""

import gradio as gr
import os
from dotenv import load_dotenv
from enhanced_recurrentgpt import EnhancedRecurrentGPT
from monitoring_dashboard import MonitoringDashboard, create_state_snapshot


# Global state
engine = None
current_state = {
    'initialized': False,
    'generating': False,
    'current_section': 0
}


def initialize_engine(initial_input, num_sections, model_name, writer_model_name):
    """Initialize the generation engine"""
    global engine, current_state

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment variables", "", None

    try:
        engine = EnhancedRecurrentGPT(
            api_key=api_key,
            model=model_name,
            writer_model=writer_model_name if writer_model_name else model_name
        )

        outline = engine.initialize_document(initial_input, int(num_sections))

        current_state['initialized'] = True
        current_state['current_section'] = 0

        outline_text = f"# {outline.title}\n\n**Goal:** {outline.goal}\n\n**Sections:**\n"
        for i, section in enumerate(outline.sections):
            outline_text += f"{i+1}. {section.title}\n   - {section.purpose}\n"

        status = f"✓ Initialized: {len(outline.sections)} sections planned"

        return status, outline_text, gr.update(interactive=True)

    except Exception as e:
        return f"Error: {str(e)}", "", None


def generate_next_section_ui(manual_instruction):
    """Generate next section (UI wrapper)"""
    global engine, current_state

    if not engine or not current_state['initialized']:
        return "Error: Engine not initialized", "", "", "", "", ""

    try:
        current_state['generating'] = True

        section_idx = current_state['current_section']

        if section_idx >= len(engine.planner.document_outline.sections):
            return "✓ Document complete!", "", "", "", "", engine.export_document()

        instruction = manual_instruction if manual_instruction.strip() else None

        result = engine.generate_next_section(section_idx, instruction)

        current_state['current_section'] += 1
        current_state['generating'] = False

        # Format results
        status = f"Section {section_idx + 1} generated"
        if result.success:
            status += f" ✓ (Quality: {result.quality_score:.1f}/10, Critique: {result.critique_score}/10)"
        else:
            status += f" ⚠ (Quality: {result.quality_score:.1f}/10, Issues: {len(result.issues)})"

        section_title = engine.planner.document_outline.sections[section_idx].title

        # Quality details
        quality_text = f"Overall: {result.quality_score:.1f}/10\n"
        quality_text += f"Critique: {result.critique_score}/10\n"
        quality_text += f"Verification: {'✓ Passed' if result.verification_passed else '✗ Failed'}\n"
        quality_text += f"Attempts: {result.attempts}"

        # Issues
        issues_text = "\n".join(result.issues) if result.issues else "None"

        # Critique feedback
        critique = result.metadata.get('critique', {})
        critique_text = f"**Strengths:**\n{critique.get('strengths', 'N/A')}\n\n"
        critique_text += f"**Weaknesses:**\n{critique.get('weaknesses', 'N/A')}\n\n"
        critique_text += f"**Suggestions:**\n{critique.get('improvement_suggestions', 'N/A')}"

        # Current document
        current_doc = engine.export_document()

        return status, section_title, result.section_text, quality_text, issues_text, critique_text, current_doc

    except Exception as e:
        current_state['generating'] = False
        return f"Error: {str(e)}", "", "", "", "", "", ""


def get_dashboard_state():
    """Get current state for dashboard"""
    global engine

    if not engine:
        return "Not initialized", 0, 0, 0, "", 0, 0, 0

    snapshot = engine.get_state_snapshot()

    return (
        snapshot.get('status', 'Unknown'),
        snapshot.get('progress', 0),
        snapshot.get('current_quality_score', 0),
        snapshot.get('average_quality_score', 0),
        snapshot.get('outline', ''),
        snapshot.get('coherence_score', 0),
        snapshot.get('vocabulary_score', 0),
        snapshot.get('structure_score', 0)
    )


def export_document_ui(format_choice):
    """Export document in chosen format"""
    global engine

    if not engine:
        return None

    try:
        content = engine.export_document(format=format_choice.lower())

        # Save to temporary file
        ext = {'Markdown': 'md', 'Text': 'txt', 'JSON': 'json', 'HTML': 'html'}
        filename = f"generated_document.{ext.get(format_choice, 'txt')}"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        return filename

    except Exception as e:
        print(f"Export error: {e}")
        return None


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title="Enhanced RecurrentGPT", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Enhanced RecurrentGPT")
        gr.Markdown("Intelligent document generation with quality monitoring and adaptive control")

        with gr.Tabs():
            # Setup Tab
            with gr.Tab("Setup"):
                gr.Markdown("## Initialize Document Generation")

                with gr.Row():
                    with gr.Column():
                        initial_input = gr.Textbox(
                            label="Topic / Initial Input",
                            placeholder="Enter the topic or initial prompt for your document...",
                            lines=3
                        )
                        num_sections = gr.Slider(
                            minimum=3, maximum=20, value=10, step=1,
                            label="Number of Sections"
                        )

                    with gr.Column():
                        model_name = gr.Dropdown(
                            choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                            value="gpt-4",
                            label="Model (for planning & critique)"
                        )
                        writer_model = gr.Dropdown(
                            choices=["", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                            value="",
                            label="Writer Model (leave empty to use same)"
                        )

                init_btn = gr.Button("Initialize Document", variant="primary", size="lg")

                init_status = gr.Textbox(label="Status", interactive=False)
                outline_display = gr.Textbox(label="Document Outline", lines=15, interactive=False)

            # Generation Tab
            with gr.Tab("Generate"):
                gr.Markdown("## Section-by-Section Generation")

                generate_btn = gr.Button("Generate Next Section", variant="primary", size="lg", interactive=False)

                manual_instruction_input = gr.Textbox(
                    label="Manual Instruction (optional)",
                    placeholder="Override planned instruction with your own...",
                    lines=2
                )

                gen_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    with gr.Column():
                        section_title_display = gr.Textbox(label="Section Title", interactive=False)
                        section_content_display = gr.Textbox(label="Generated Content", lines=12, interactive=False)

                    with gr.Column():
                        quality_display = gr.Textbox(label="Quality Metrics", lines=4, interactive=False)
                        issues_display = gr.Textbox(label="Issues", lines=4, interactive=False)
                        critique_display = gr.Textbox(label="Critique Feedback", lines=8, interactive=False)

            # Document View Tab
            with gr.Tab("Document"):
                gr.Markdown("## Current Document")

                refresh_doc_btn = gr.Button("🔄 Refresh", size="sm")

                document_display = gr.Textbox(
                    label="Generated Document",
                    lines=25,
                    interactive=False
                )

                with gr.Row():
                    format_choice = gr.Radio(
                        choices=["Markdown", "Text", "JSON", "HTML"],
                        value="Markdown",
                        label="Export Format"
                    )
                    export_btn = gr.Button("📥 Export Document", variant="secondary")

                export_file = gr.File(label="Download")

            # Dashboard Tab
            with gr.Tab("Dashboard"):
                gr.Markdown("## Monitoring Dashboard")

                refresh_dash_btn = gr.Button("🔄 Refresh Dashboard")

                with gr.Row():
                    status_display = gr.Textbox(label="Status", interactive=False)
                    progress_display = gr.Slider(minimum=0, maximum=100, value=0, label="Progress (%)", interactive=False)

                with gr.Row():
                    current_quality_display = gr.Number(label="Current Quality", value=0, interactive=False)
                    avg_quality_display = gr.Number(label="Average Quality", value=0, interactive=False)

                outline_dash = gr.Textbox(label="Document Outline", lines=10, interactive=False)

                gr.Markdown("### Quality Metrics")
                with gr.Row():
                    coherence_display = gr.Number(label="Semantic Coherence", value=0, interactive=False)
                    vocabulary_display = gr.Number(label="Vocabulary Consistency", value=0, interactive=False)
                    structure_display = gr.Number(label="Structural Balance", value=0, interactive=False)

        # Event handlers
        init_btn.click(
            fn=initialize_engine,
            inputs=[initial_input, num_sections, model_name, writer_model],
            outputs=[init_status, outline_display, generate_btn]
        )

        generate_btn.click(
            fn=generate_next_section_ui,
            inputs=[manual_instruction_input],
            outputs=[
                gen_status, section_title_display, section_content_display,
                quality_display, issues_display, critique_display, document_display
            ]
        )

        refresh_doc_btn.click(
            fn=lambda: engine.export_document() if engine else "",
            outputs=document_display
        )

        export_btn.click(
            fn=export_document_ui,
            inputs=format_choice,
            outputs=export_file
        )

        refresh_dash_btn.click(
            fn=get_dashboard_state,
            outputs=[
                status_display, progress_display, current_quality_display,
                avg_quality_display, outline_dash, coherence_display,
                vocabulary_display, structure_display
            ]
        )

    return app


if __name__ == '__main__':
    app = create_interface()
    print("\nLaunching Enhanced RecurrentGPT Interface...")
    print("Access at: http://localhost:7860\n")
    app.launch(server_port=7860, share=False)
