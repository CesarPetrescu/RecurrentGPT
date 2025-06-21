import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from openai_embedder import OpenAIEmbedder
from utils import get_init, parse_instructions
import re


_CACHE = {}


# Build the semantic search model
embedder = OpenAIEmbedder()

def init_prompt(doc_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
Please write a {doc_type} research document{description} focused on drone technology. Follow the format below precisely:

Begin with the title of the document.
Next, provide an outline for the introduction section describing the background and motivation.
Write the first three sections based on your outline. Use an informative tone and elaborate on drone-related topics.
Write a summary that captures the key information of these sections.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible informative continuation of the document.
The output format should follow these guidelines:
Title: <title of the document>
Outline: <outline for the introduction>
Section 1: <content for section 1>
Section 2: <content for section 2>
Section 3: <content for section 3>
Summary: <content of summary>
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly.

"""

def init(doc_type, description, request: gr.Request):
    if doc_type == "":
        doc_type = "Technical"
    global _CACHE
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    # prepare first init
    init_paragraphs = get_init(text=init_prompt(doc_type,description))
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""Title: {init_paragraphs['name']}

Outline: {init_paragraphs['Outline']}

Sections:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(short_memory, long_memory, instruction1, instruction2, instruction3, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        #randomly select one instruction out of three
        instruction_index = random.randint(0,2)
        output['output_instruction'] = [instruction1, instruction2, instruction3][instruction_index]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(short_memory, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], parse_instructions(writer.long_memory), current_paras + '\n\n' + writer.output['input_paragraph'], *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan

with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    with gr.Tab("Auto-Generation"):
        with gr.Column():
            with gr.Row():
                doc_type = gr.Textbox(
                    label="Document Type", placeholder="e.g. technical")
                description = gr.Textbox(label="Topic")
            btn_init = gr.Button(
                "Init Document Generation", elem_id="init_button")
            gr.Examples(["Technical", "Whitepaper", "Academic", "Proposal", "Survey"],
                        inputs=[doc_type], elem_id="example_selector")
            written_paras = gr.Textbox(
                label="Written Sections (editable)", lines=21)

        with gr.Column():
            gr.Markdown("### Memory Module")
            short_memory = gr.Textbox(
                label="Short-Term Memory (editable)", lines=3)
            long_memory = gr.Textbox(
                label="Long-Term Memory (editable)", lines=6)
            gr.Markdown("### Instruction Module")
            instruction1 = gr.Textbox(
                label="Instruction 1 (editable)", lines=4)
            instruction2 = gr.Textbox(
                label="Instruction 2 (editable)", lines=4)
            instruction3 = gr.Textbox(
                label="Instruction 3 (editable)", lines=4)
            selected_plan = gr.Textbox(
                label="Revised Instruction (from last step)", lines=2)

        btn_step = gr.Button("Next Step", elem_id="step_button")
        btn_init.click(init, inputs=[doc_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])

    demo.launch()

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True,
                server_name="0.0.0.0", show_api=False)