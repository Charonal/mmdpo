import openai
from data_process import read_gsm8k,read_gsmhard
from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
from vllm import LLM, SamplingParams
from utils import batch_prompt,build_batch_datas,map_output,extract_output,build_step2step_prompt,batch_step2step_prompt





def main():
    file = '/data/lyz/math_dpo/datas/step2step/gsm8k_step2step_sample.json'
    output_file = '/data/lyz/math_dpo/datas/step2step/gsm8k_step2step_gen_k.json'
    with open(file) as f:
        datas_select = json.load(f)
    print('total len: ',len(datas_select))

    llm = LLM(model="/data/lyz/qwen/Qwen2-7B-Instruct", trust_remote_code=True, tensor_parallel_size=2,tokenizer_mode="auto", dtype="auto",gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.9,max_tokens=512)
    start_time = time.time()
    datas_final =[]
    batch_size = 10
    gen_nums = 5
    # for data in tqdm(datas_select):
    #     batch = build_step2step_prompt(data)
    #     outputs = llm.chat(batch,
    #                     sampling_params=sampling_params,
    #                     use_tqdm=False)

    #     data['step2step_gen'] = extract_output(outputs)
    #     datas_final.append(data)
    batch_datas = build_batch_datas(datas_select,batch_size)

    try:
        for data in tqdm(batch_datas):
            # print('len data: ',len(data))
            batch = batch_step2step_prompt(data,gen_nums)
            # print('len batch: ',len(batch))
            try:
                outputs = llm.chat(batch,
                                sampling_params=sampling_params,
                                use_tqdm=False)
                # print(len(outputs))
                batch = map_output(data,outputs = outputs,gen_nums=gen_nums)
                datas_final += batch
            except Exception as e:
                print(e)
    finally:
        del llm
    end_time = time.time()
    print('total time spend: ',end_time - start_time)
    with open(output_file,'w') as f:
        json.dump(datas_final,f,indent = 4)
    print('total datas_final len : ',len(datas_final))
if __name__ == '__main__':
    main()