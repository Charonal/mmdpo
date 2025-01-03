#原始gsm8k数据读取
import json
import jsonlines
import re
import random

# 读取 jsonl 文件并添加 final_answer 字段
def read_gsm8k(jsonl_file):
    datas = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            answer_text = data.get("answer", "")
            
            # 提取 '####' 后面的答案并转换为数字类型
            if '####' in answer_text:
                final_answer = answer_text.split('####')[-1].strip()
                try:
                    data["final_answer"] = float(final_answer)  # 新增 final_answer 字段
                except ValueError:
                    data["final_answer"] = None  # 无法转换时设为 None
            
            # 写入处理后的数据
            datas.append(data)
    return datas

def read_gsmhard(file):
    datas = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data['question'] = data['input']
            data['final_answer'] = data['target']
            datas.append(data)
    return datas


def extract_numerical_answer(solutions,target):
    answers = []
    for i,solution in enumerate(solutions):
        # 使用正则表达式提取 "So, the numerical answer is:" 后的数字
        match = re.search(r"So, the numerical answer is:(.*)", solution, re.IGNORECASE | re.DOTALL)

        if match:
            # 提取出这部分文本内容
            answer_line = match.group(1).strip()
        else:
            answer_line = solution.split('\n')[-1]
            # 从中提取数字（包括整数和小数）
        num_match = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_line)

        if num_match:
            flag = 0
            for num in num_match:
                final_answer_str = num.replace(',', '').replace(' ', '')
                try:
                    # 转换为浮点数或整数
                    final_answer = float(final_answer_str) if '.' in final_answer_str else int(final_answer_str)
                    if final_answer == target:
                        answers.append(final_answer)
                        flag = 1 
                        break
                except ValueError:
                    print('error')
                    answers.append(None)  # 如果转换失败设为 None
            if flag == 0:
                final_answer = float(final_answer_str) if '.' in final_answer_str else int(final_answer_str)
                answers.append(final_answer)
        else:
            answers.append(None)
    return answers

def eval_answer(answers,target):
    labels = []
    for ans in answers:
        if not ans:
            labels.append(None)
        else:
            labels.append(float(ans) == target)
    return labels

def split_and_sample_steps(text):
    # 检查是否存在 [Step ] 或 [step ] 标签
    if re.search(r"\[Step\s*\d+\]", text, re.IGNORECASE):
        # 使用 [Step ] 或 [step ] 标签进行分割
        steps = re.split(r"\[Step\s*\d+\]:?", text, flags=re.IGNORECASE)
    else:
        # 如果没有找到 [Step ] 标签，用换行符 \n\n 或 \n 分割
        steps = re.split(r"\n\n|\n", text)
    
    # 移除空白步骤并去除每个步骤的前后空格
    steps = [step.strip() for step in steps if step.strip()]
    
    # 获取实际步骤数
    step_count = len(steps)
    
    # 随机生成一个 k 值，在 1 到 step_count 范围内
    k = random.randint(0, step_count-1)
    
    # 返回前 k 个步骤
    sampled_steps = steps[:k]
    restored_steps = [f"[Step {i + 1}]:\n{step}" for i, step in enumerate(sampled_steps)]
    
    # 将恢复后的步骤组合成完整文本
    restored_text = "\n\n".join(restored_steps)
    return restored_text


# def dpo_fomat():
#     format = {
#         'instruct':instruct,
#         'chosen':chosen,
#         'reject':reject,
#     }
# def build_sol2sol_data():
#     return datas
# def build_innerstep_data():
#     return datas
# def build_step2step_data():
#     return datas