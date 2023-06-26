import openai
import json
import time
from tqdm import tqdm
import pandas as pd
import utils

db = pd.read_csv('./data/jobs_salary.csv')

def trace_database(protocol):
    """
    Get results from the database based on the protocol.

    Args:
        protocol: the key components of a question

    Returns:
        Statistics information.
    """
    # TODO: replace with the code from Aghil and Adrian


    row = db[(db['Location'] == protocol['Location']) &
             (db['Company'] == protocol['Company'].lower()) &
             (db['Title'] == protocol['Title']) &
             (db['Level'] == protocol['Level'])]

    median = row.iloc[0]['median']
    q1 = row.iloc[0]['q1']
    q3 = row.iloc[0]['q3']
    
    return median, q1, q3

def generate_prompt(protocol, num_question):
    """
    Convert protocol to a full prompt used for generating question list.

    Args:
        protocol: the key components of a question
        num_question: total number of questions to generate

    Returns:
        Prompt.
    """
    location = protocol['Location']
    company  = protocol['Company']
    title    = protocol['Title']
    level    = protocol['Level']

    with open('prompt_template.txt', 'r') as f:
        template = f.read()

    prompt = template.format(
        num = num_question,
        location = location,
        company = company,
        title = title,
        level = level
    )

    # return the prompt
    return prompt

def make_requests(prompt):
    """
    Generate question list via OpenAI API call.
    
    Args:
        prompt: the full prompt for generating question list

    Returns:
        The generated question list
    """

    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}
                ]
            )
            res = response.choices[0].message.content
            question_list = res.split('\n')
            question_list = [q.strip() for q in question_list if q.strip() != '']

            if len(question_list) > 0:
                break
            else:
                continue

        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            time.sleep(2)

    return question_list

def generate_question(protocol, num_question):
    """
    Generate question list based on protocol.

    Args:
        protocol: the key components of a question
        num_question: total number of questions to generate

    Returns:
        List of questions based on protocol.
    """
    # TODO: implement function to ensure len(question_list) == num_question
    prompt = generate_prompt(protocol, num_question)
    question_list = make_requests(prompt)

    return question_list

def generate_answer(protocol):
    """
    Generate formatted answer based on protocol.

    Args:
        protocol: the key components of a question

    Returns:
        Formatted answer.
    """

    median, q1, q3 = trace_database(protocol)

    with open('answer_template.txt', 'r') as f:
        answer_template = f.read()

    answer = answer_template.format(
        median = median,
        q1 = q1,
        q3 = q3,
        location = protocol['Location'],
        company = protocol['Company'],
        title = protocol['Title'],
        level = protocol['Level']
    )

    return answer

def generate_data_point(protocol, num_question):
    """
    Combine questions and answers to generate data for fine-tuning.

    Args:
        protocol: the key components of a question
        num_questions: total number of questions to generate

    Returns:
        Question-answer pairs.
    """
    question_list = generate_question(protocol, num_question)
    answer = generate_answer(protocol)

    res = [
        {
            'instruction': question_list[i],
            'input': '',
            'output': answer
        }
        for i in range(len(question_list))
    ]

    return res

def generate_dataset(protocol_list, num_question=3):
    dataset = []
    for protocol in tqdm(protocol_list):
        dataset += generate_data_point(protocol, num_question)
        
    return dataset

def generate_train_test_dataset(protocol_list, num_train=7, num_test=3):
    train_dataset = []
    test_dataset = []
    for i in range(len(protocol_list)):
        protocol = protocol_list[i]
        train_dataset += generate_data_point(protocol, num_train)
        test_dataset += generate_data_point(protocol, num_test)
        print('{}/{} completed'.format(i + 1, len(protocol_list)))
        
    return train_dataset, test_dataset

if __name__ == '__main__':
    
    # replace this with the directory of your API key
    with open('../credential/api_key.txt', 'r') as f:
        api = f.readline().strip()
    openai.api_key = api

    with open('jobs_protocol.json', 'r') as f:
    # with open('protocol_list.json', 'r') as f:
        protocol_list = json.load(f)

    # dataset = generate_dataset(protocol_list, num_question=10)

    # with open('fine_tuning_dataset.json', 'w') as f:
    #     json.dump(dataset, f, indent=4)

    train_dataset, test_dataset = generate_train_test_dataset(protocol_list, 7, 3)

    with open('fine_tuning_train_dataset.json', 'w') as f:
        json.dump(train_dataset, f, indent=4)

    with open('fine_tuning_test_dataset.json', 'w') as f:
        json.dump(test_dataset, f, indent=4)
    