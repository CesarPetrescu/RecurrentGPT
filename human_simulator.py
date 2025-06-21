
from utils import get_content_between_a_b, parse_instructions,get_api_response

class Human:

    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input['output_memory']
        self.embedder = embedder
        self.output = {}


    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = f"""
        Now imagine you are an AI-powered document generator specializing in drone technology. You will be given a previously written section (written by you), a section written by your assistant, a summary of the main drone-related topics maintained by your assistant, and a plan for the next section proposed by your assistant.

    I need you to write:
    1. Extended Section: Expand the new section written by the assistant to twice its original length.
    2. Selected Plan: Copy the plan proposed by your assistant.
    3. Revised Plan: Refine the selected plan into a concise outline for the next section.

    Previously written section:
    {previous_paragraph}

    Summary of the main drone topics maintained by your assistant:

    {memory}

    New section written by your assistant:
    {writer_new_paragraph}

    Plan for the next section proposed by your assistant:
    {user_edited_plan}

    Now start writing, organize your output by strictly following the output format as below:

    Extended Section:
    <string of output section>, around 40-50 sentences.

    Selected Plan:
    <copy the plan here>

    Revised Plan:
    <string of revised plan>, keep it short, around 5-7 sentences.

    Very Important:
    Remember that you are drafting a formal document on drone technology. Write in an informative style and do not move too quickly when outlining the next section. Think about how the plan can remain engaging and coherent. Remember to follow the length constraints! The document will grow over many sections, so leave room for future content.


    """
        return input_text
    
    def parse_plan(self,response):
        plan = get_content_between_a_b('Selected Plan:','Reason',response)
        return plan


    def select_plan(self,response_file):
        
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        prompt = f"""
    Now imagine you are a helpful assistant who supports a researcher writing a document about drone technology. You will be given a previously written section and a section written by your assistant, a summary of the main drone topics maintained by the assistant, and 3 different possible plans of what to write next.
    I need you to:
    Select the most informative and suitable plan proposed by the assistant.

    Previously written section:
    {previous_paragraph}

    Summary of the main drone topics maintained by the assistant:
    {memory}

    The new section written by your assistant:
    {writer_new_paragraph}

    Three plans of what to write next proposed by your assistant:
    {parse_instructions(previous_plans)}

    Now start choosing, organize your output by strictly following the output format as below:
      
    Selected Plan: 
    <copy the selected plan here>

    Reason:
    <Explain why you choose the plan>
    """
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        plan = self.parse_plan(response)
        while plan == None:
            response = get_api_response(prompt)
            plan= self.parse_plan(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Selected plan here:\n{response}\n\n")

        return plan
        
    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith('Extended Section'):
                new_paragraph = get_content_between_a_b(
                    'Extended Section:', 'Selected Plan', text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != '\n' and lines[-1].startswith('Revised Plan:'):
                revised_plan = lines[-1][len("Revised Plan:"):]
            elif lines[-1] != '\n':
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                # "selected_plan": selected_plan,
                "output_instruction": revised_plan,
                # "memory":self.input["output_memory"]
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
                f.write(f"Human's output here:\n{response}\n\n")
