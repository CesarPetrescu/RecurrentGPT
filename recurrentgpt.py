from utils import get_content_between_a_b, get_api_response
import random
from openai_embedder import cosine_scores


class RecurrentGPT:

    def __init__(self, input, short_memory, long_memory, memory_index, embedder):
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        if self.long_memory and not memory_index:
            self.memory_index = self.embedder.encode(self.long_memory)
        else:
            self.memory_index = memory_index
        self.output = {}

    def prepare_input(self, new_character_prob=0.1, top_k=2):

        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]

        instruction_embedding = self.embedder.encode(input_instruction)[0]

        # get the top 3 most similar paragraphs from memory

        memory_scores = cosine_scores(
            instruction_embedding, self.memory_index)
        top_k_idx = sorted(range(len(memory_scores)),
                           key=lambda i: memory_scores[i],
                           reverse=True)[:top_k]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]
        # combine the top 3 paragraphs
        input_long_term_memory = '\n'.join(
            [f"Related Paragraphs {i+1} :" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the output paragrah and add it into the memory."
        else:
            new_character_prompt = ""

        input_text = f"""I need you to help me write a research document about drones. I will provide a memory (a brief summary) of 400 words to store key content so you can keep track of very long context. Each time, I will give you the current memory, a previously written section, and instructions on what to write in the next section.
    I need you to write:
    1. Output Section: the next section of the document. The output section should contain around 20 sentences and should follow the input instructions.
    2. Output Memory: The updated memory. You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that you should write the updated memory. The updated memory should be similar to the input memory except the parts you previously thought that should be deleted or added. The updated memory should only store key information. The updated memory should never exceed 20 sentences!
    3. Output Instruction: instructions of what to write next (after what you have written). You should output 3 different instructions, each describing a possible informative continuation. Each output instruction should contain around 5 sentences
    Here are the inputs: 

    Input Memory:  
    {self.short_memory}

    Input Section:
    {input_paragraph}

    Input Instruction:
    {input_instruction}

    Input Related Paragraphs:
    {input_long_term_memory}
    
    Now start writing, organize your output by strictly following the output format as below:
    Output Section:
    <string of output section>, around 20 sentences.

    Output Memory: 
    Rational: <string that explain how to update the memory>;
    Updated Memory: <string of updated memory>, around 10 to 20 sentences

    Output Instruction: 
    Instruction 1: <content for instruction 1>, around 5 sentences
    Instruction 2: <content for instruction 2>, around 5 sentences
    Instruction 3: <content for instruction 3>, around 5 sentences

    Very important!! The updated memory should only store key information. The updated memory should never contain over 500 words!
    Finally, remember that you are writing a technical document about drone technology. Write in an informative style and do not move too fast when drafting instructions for the next section. The document will contain many sections, so leave room for future content. Think about how the document can remain engaging and coherent when providing the next instructions.

    Very Important: 
    You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that, you start rewrite the input memory to get the updated memory. 
    {new_character_prompt}
    """
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                'Output Section:', 'Output Memory', output)
            output_memory_updated = get_content_between_a_b(
                'Updated Memory:', 'Output Instruction:', output)
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b(
                'Instruction 1:', 'Instruction 2', output)
            ins_2 = get_content_between_a_b(
                'Instruction 2:', 'Instruction 3', output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
                ins_3 = lines[-1][len("Instruction 3:"):]
            elif lines[-1] != '\n':
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "output_paragraph": output_paragraph,
                "output_instruction": [instruction.strip() for instruction in output_instructions]
            }

            return output
        except:
            return None

    def step(self, response_file=None):

        prompt = self.prepare_input()

        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Writer's output here:\n{response}\n\n")

        self.long_memory.append(self.input["output_paragraph"])
        self.memory_index = self.embedder.encode(self.long_memory)
