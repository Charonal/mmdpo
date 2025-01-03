import re
import json
import csv
from collections import defaultdict
import random
def get_template():
    template = """
    {% if messages|length == 2 and messages[0]['role'] == 'system' and messages[1]['role'] == 'user' %}
    <|im_start|>system
    {{ messages[0]['content'] }}
    <|im_end|>
    <|im_start|>human
    {{ messages[1]['content'] }}
    <|im_end|>
    <|im_start|>assistant
    {% else %}
    {% for message in messages %}
    <|im_start|>{{ message['role'] }}
    {{ message['content'] }}
    <|im_end|>
    {% endfor %}
    {% endif %}
    """
    return template

### prompt 构建 
def build_prompt(data):
    question = data['question']
    system_prompt = """You are a helpful assistant. You must follow the following rules:
                        1. You need to help me to solve the math problem step by step.
                        2. Use the prefix [Step x]: to indicate each step. x is the index of the current step.
                        3. In each [Step x], you can only include one computational formula, which means you need to divide the answer into as many steps as possible.
                        4. And at the end of the output, you should repeat your final numerical answer use \'So, the numerical answer is: [number]\'. [number] is the final numerical answer. And does not include any other explanatory notes.
                        
                        For example:
                            <question>
                            Let's think step by step.
                            [Step 1]: ...
                            [Step 2]: ...
                            ...
                            [Step n]: ...
                            So, the numerical answer is: [number]
                    """
    # instruct = "You should solve the math problem step by step. And at the end of the output, you should give your final numerical answer use \'So, the numerical answer is: \'.\n"
    gen_prompt = "Let's think step by step.\n [Step 1]: "
    prompt = '<question>: '+ question + gen_prompt
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

### infer2infer prompt 
def build_infer2infer_prompt(data,gen_num):
    question = data['question']
    combined_steps = data['combined_steps']
    
    system_prompt = """You are a helpful assistant.
                        1. You need to continue writing based on the math problem I provided and the unfinished steps to get the final answer. 
                        2. Use the prefix [Step x]: to indicate each step. X is further numbered based on the number of steps given in the previous text.
                        3. And at the end of the output, you should repeat your final numerical answer use \'So, the numerical answer is: [number]\'. [number] is the final numerical answer. And does not include any other explanatory notes.
                   """
    messages = []
    for steps in combined_steps:
        prompt = question + "Let's think step by step.\n" + steps
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        messages += [message for _ in range(gen_num)]
    return messages

### step2step prompt 构建
def build_step2step_prompt(data):
    question = data['question']
    front_step = data['front_step']
    false_k_step = data['false_k_step']
    k_step = data['k_step']
    system_prompt = """You are a helpful assistant.
                        1. You need to continue writing based on the math problem I provided and the unfinished steps to get the final answer. 
                        2. Use the prefix [Step x]: to indicate each step. X is further numbered based on the number of steps given in the previous text.
                        3. And at the end of the output, you should repeat your final numerical answer use \'So, the numerical answer is: [number]\'. [number] is the final numerical answer. And does not include any other explanatory notes.
                    """
    gen_prompt = "Let's think step by step.\n [Step 1]: "
    prompt = '<question>: '+ question + gen_prompt + front_step + k_step
    messages = []
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    # messages += [message for _ in range(5)]
    # return messages
    return message



def batch_prompt(batch_data,gen_nums):
    batch_prompt = []
    for data in batch_data:
        message = build_prompt(data)
        batch_prompt += [message for _ in range(gen_nums)]
    return batch_prompt

def batch_step2step_prompt(batch_data,gen_nums):
    batch_prompt = []
    for data in batch_data:
        message = build_step2step_prompt(data)
        batch_prompt += [message for _ in range(gen_nums)]
    return batch_prompt

def extract_output(outputs):
    extract_output = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        extract_output.append(generated_text)
    return extract_output

def map_output(batch,outputs,gen_nums):
    outputs = extract_output(outputs)
    for i,data in enumerate(batch):
        data['solutions'] = outputs[i*gen_nums:(i+1)*gen_nums]
    return batch

def build_batch_datas(datas,batch_size):
    cnt = 0
    batch_datas = []
    batch = []
    for data in datas:
        if len(batch) == batch_size:
            batch_datas.append(batch)
            batch = []
        batch.append(data)
    return batch_datas


def acc_eval(datas,mode):
    cnt = 0
    for data  in datas:
        is_answer = data['is_answer']
        if mode == 'top1':
            if is_answer[0]:
                cnt += 1 
        elif mode == 'all':
            for i in is_answer:
                if i:
                    cnt += 1 
                    
    acc = cnt / len(datas)        
    return acc

def build_test_prompt(data):
    question = data['question']
    instruct = """
    You need to continue writing based on the math problem I provided and the front_steps to get the final answer. 

    math problem: {question}
    front_steps:{front_steps}
    """

    system_prompt = """You are a helpful assistant.
                        1. You need to continue writing based on the math problem I provided and the front_steps to get the final answer. 
                        2. Use the prefix [Step x]: to indicate each step. X is further numbered based on the number of steps given in the previous text.
                        3. And at the end of the output, you should repeat your final numerical answer use \'So, the numerical answer is: [number]\'. [number] is the final numerical answer. And does not include any other explanatory notes.
                   """
    front_step = "Let's think step by step.\n"
    prompt = instruct.format(question=question,front_steps = front_step)
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

def batch_test_prompt(batch_data,gen_nums):
    batch_prompt = []
    for data in batch_data:
        message = build_test_prompt(data)
        batch_prompt += [message for _ in range(gen_nums)]
    return batch_prompt

def build_formula_prompt(data):
    question = data['mask_question']
    instruct = """
    math problem: {question}
    Let's think step by step.\n
    """

    system_prompt = """You are a helpful assistant.
                    1. You need to reason based on the math application problem I gave you.
                    2. The math application problem I gave did not involve specific numbers, but instead used [num_i] instead, so you only need to reason out a formula to get the final result, such as answer=[num_1]+[num_2]/[num_3];
                    3. At the end of the output, you need to use 'So, The formula is: [formula] 'to output the results, which is convenient for me to extract. For example, 'So, the formula is: answer = [num_1]+[num_2] / [num_3]';
                    """
    front_step = "Let's think step by step.\n"
    prompt = instruct.format(question=question)
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

def batch_formula_prompt(batch_data,gen_nums):
    batch_prompt = []
    for data in batch_data:
        message = build_formula_prompt(data)
        batch_prompt += [message for _ in range(gen_nums)]
    return batch_prompt