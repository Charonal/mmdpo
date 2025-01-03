import json
import re
from utils import build_test_prompt,extract_output,build_batch_datas,map_output,batch_test_prompt
from data_process import read_gsm8k,read_gsmhard
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

def main():
    file = '/data/lyz/math_dpo/datas/gsm8k/test.jsonl'
    model = 'sol_infer_without_sft_hp0_10'
    output_file = '/data/lyz/math_dpo/output/gsm8k_test_{model}.json'.format(model = model)
    if model == 'qwen':
        model_path = '/data/lyz/qwen/Qwen2-7B-Instruct'
    elif model =='step_dpo':
        model_path = '/data/lyz/math_dpo/merge_models/merge_model_step_dpo'
    elif model == 'simpo':
        model_path= "/data/lyz/math_dpo/merge_models/merge_model_sol_infer"
    else:
        model_path = '/data/lyz/math_dpo/merge_models/merge_model_'+model

    datas = read_gsm8k(file)
    batch_size = 20
    gen_nums = 5
    llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=2,tokenizer_mode="auto", dtype="auto",gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9,max_tokens=512)
    datas_final =[]
    batch_datas = build_batch_datas(datas,batch_size)
    for data in tqdm(batch_datas):
        batch = batch_test_prompt(data,gen_nums)
        outputs = llm.chat(batch,
                        sampling_params=sampling_params,
                        use_tqdm=False)
        print(len(outputs))
        batch = map_output(data,outputs = outputs,gen_nums=gen_nums)
        datas_final += batch

    with open(output_file,'w') as f:
        json.dump(datas_final,f,indent = 4)


if __name__ == '__main__':
    main()