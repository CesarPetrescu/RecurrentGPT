from recurrentgpt import RecurrentGPT
from human_simulator import Human
import json
import argparse
from openai_embedder import OpenAIEmbedder
from utils import get_init

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ChatGPT-based automatic novel writing')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--r_file', type=str, default='response.txt')
    parser.add_argument('--init_prompt', type=str, default='init_prompt.json')
    parser.add_argument('--type', type=str, default='science fiction')
    parser.add_argument('--topic', type=str, default='')
    args = parser.parse_args()

    prompts = json.load(open(args.init_prompt, 'r'))
    init_prompt = prompts['init_prompt'].format(type=args.type, topic=args.topic)

    init_paragraphs = get_init(init_text=None, text=init_prompt, response_file=args.r_file)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n'.join([
            init_paragraphs['Paragraph 1'],
            init_paragraphs['Paragraph 2']]),
        'output_memory': init_paragraphs['Summary'],
        'output_instruction': [
            init_paragraphs['Instruction 1'],
            init_paragraphs['Instruction 2'],
            init_paragraphs['Instruction 3']]
    }

    embedder = OpenAIEmbedder()
    human = Human(input=start_input_to_human, memory=None, embedder=embedder)
    human.input['output_instruction'] = human.select_plan(args.r_file)
    print(human.input['output_instruction'])
    human.step(args.r_file)
    start_short_memory = init_paragraphs['Summary']
    writer_start_input = human.output

    writer = RecurrentGPT(
        input=writer_start_input,
        short_memory=start_short_memory,
        long_memory=[init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']],
        memory_index=None,
        embedder=embedder)

    for _ in range(args.iter):
        writer.step(args.r_file)
        human.input = writer.output
        human.input['output_instruction'] = human.select_plan(args.r_file)
        human.step(args.r_file)
        writer.input = human.output

