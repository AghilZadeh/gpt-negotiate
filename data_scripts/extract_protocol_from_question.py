import openai
import json
from utils import us_state_to_abbrev
import re

def load_questions(datapath='./data/fine_tuning_train_dataset.json') -> list:
    "loads all the questions from datapath"

    # loading data (questions)
    with open(datapath, 'r') as f:
        protocol_list = json.load(f)
    questions = [protocol_list[i]['instruction'][3:] for i in range(len(protocol_list))]
    return questions


def load_answers(datapath='./data/jobs_protocol.json') -> list:
    "creates all the true answer protocols for all the questions (len(questions)=7*len(protocols))"

    # loading data (true labels)
    with open(datapath, 'r') as f:
        answer_list = json.load(f)
    answers = [answer_list[i//7] for i in range(len(answer_list)*7)]
    return(answers)


def generate_answer(question:str) -> dict: 
    "generates an answer protocole by querying openai API"

    # replace this with the directory of your API key
    with open('./credential/api_key.txt', 'r') as f:
        api = f.readline().strip()
    openai.api_key = api

    system_instruction = """
                        Put the user's message into a JSON with keys "Location", "Company", "Title" and "Level" and strictly avoid any other description.
                    ### instruction: 
                    1 - Title is either Software Engineer or Data Scientist.
                    2 - If you could not find the value of the key, return N/A
                    3 - Just provide the JSON string without any other text and double quotes around the keys.
                    """
    
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": question}
                ]
            )
    response_content = response['choices'][0]['message']['content']
    json_content = re.search(r"{(.|\n)*?}", response_content).group()

    # extracted information from the question
    generated_answer = json.loads(json_content)
    ## cleaning the generated answer to match with the true labels
    if generated_answer['Location'] in us_state_to_abbrev:
        generated_answer['Location'] = us_state_to_abbrev[generated_answer['Location']]
    generated_answer['Company'] = str.lower(generated_answer['Company'])

    return(generated_answer)


if __name__ == '__main__':
    
    questions = load_questions()
    answers = load_answers()
    i = 200
    print('\n')
    print(questions[i])
    generated_answer = generate_answer(questions[i])
    
    print('\n')
    print('Generated Answer:', generated_answer)
    print('True Answer:     ', answers[i])
    print('\n')