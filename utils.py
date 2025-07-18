import os
import re
from openai import OpenAI

# Load local configuration if available
def load_env(path="config.env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()

api_key = os.getenv("ApiKey") or os.getenv("OPENAI_API_KEY")
api_base = os.getenv("ApiBase") or os.getenv("OPENAI_API_BASE")
model_name = os.getenv("ModelName", "gpt-4o-2024-05-13")
def get_api_response(content: str, max_tokens=None):
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and creative assistant for writing research documents about drones.'
        }, {
            'role': 'user',
            'content': content,
        }],
        temperature=0.5,
        stream=True
    )
    final = ''
    for chunk in response:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                final += chunk.choices[0].delta.content
                # print(final)
    return final 

def get_content_between_a_b(a,b,text):
    return re.search(f"{a}(.*?)\n{b}", text, re.DOTALL).group(1).strip()


def get_init(init_text=None,text=None,response_file=None):
    """
    init_text: if the title, outline, and the first 3 paragraphs are given in a .txt file, directly read
    text: if no .txt file is given, use init prompt to generate
    """
    if not init_text:
        response = get_api_response(text)
        print(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Init output here:\n{response}\n\n")
    else:
        with open(init_text,'r',encoding='utf-8') as f:
            response = f.read()
        f.close()
    paragraphs = {
        "name":"",
        "Outline":"",
        "Paragraph 1":"",
        "Paragraph 2":"",
        "Paragraph 3":"",
        "Summary": "",
        "Instruction 1":"",
        "Instruction 2":"", 
        "Instruction 3":""    
    }
    paragraphs['name'] = get_content_between_a_b('Name:','Outline',response)
    
    paragraphs['Paragraph 1'] = get_content_between_a_b('Paragraph 1:','Paragraph 2:',response)
    paragraphs['Paragraph 2'] = get_content_between_a_b('Paragraph 2:','Paragraph 3:',response)
    paragraphs['Paragraph 3'] = get_content_between_a_b('Paragraph 3:','Summary',response)
    paragraphs['Summary'] = get_content_between_a_b('Summary:','Instruction 1',response)
    paragraphs['Instruction 1'] = get_content_between_a_b('Instruction 1:','Instruction 2',response)
    paragraphs['Instruction 2'] = get_content_between_a_b('Instruction 2:','Instruction 3',response)
    lines = response.splitlines()
    # content of Instruction 3 may be in the same line with I3 or in the next line
    if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
        paragraphs['Instruction 3'] = lines[-1][len("Instruction 3:"):]
    elif lines[-1] != '\n':
        paragraphs['Instruction 3'] = lines[-1]
    # Sometimes it gives Chapter outline, sometimes it doesn't
    for line in lines:
        if line.startswith('Chapter'):
            paragraphs['Outline'] = get_content_between_a_b('Outline:','Chapter',response)
            break
    if paragraphs['Outline'] == '':
        paragraphs['Outline'] = get_content_between_a_b('Outline:','Paragraph',response)


    return paragraphs

def get_chatgpt_response(model,prompt):
    response = ""
    for data in model.ask(prompt):
        response = data["message"]
    model.delete_conversation(model.conversation_id)
    model.reset_chat()
    return response


def parse_instructions(instructions):
    output = ""
    for i in range(len(instructions)):
        output += f"{i+1}. {instructions[i]}\n"
    return output
