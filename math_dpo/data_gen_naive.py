import openai
from data_process import read_gsm8k,read_gsmhard
from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
from vllm import LLM, SamplingParams
from utils import batch_prompt,build_batch_datas,map_output





def main():
    gsm8k_file = '/data/lyz/math_dpo/datas/gsm8k/test.jsonl'
    gsmhard_file = '/data/lyz/math_dpo/datas/gsmhardv2.jsonl'
    output_file = '/data/lyz/math_dpo/output/gsm8k_test_gen_naive.json'
    datas = read_gsm8k(gsm8k_file)
    # datas = read_gsmhard(gsmhard_file)
    batch_size = 6
    gen_nums = 10
    llm = LLM(model="/data/lyz/qwen/Qwen2-7B-Instruct", trust_remote_code=True, tensor_parallel_size=2,tokenizer_mode="auto", dtype="auto",gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9,max_tokens=512)
    start_time = time.time()
    datas_final =[]
    start_time = time.time()
    batch_datas = build_batch_datas(datas,batch_size)
    print('data  len:' ,len(datas))
    for data in tqdm(batch_datas):
        # print('len data: ',len(data))
        batch = batch_prompt(data,gen_nums)
        # print('len batch: ',len(batch))
        outputs = llm.chat(batch,
                        sampling_params=sampling_params,
                        use_tqdm=False)
        # print(len(outputs))
        batch = map_output(data,outputs = outputs,gen_nums=gen_nums)
        datas_final += batch
    end_time = time.time()
    print('total time spend: ',end_time - start_time)
    with open(output_file,'w') as f:
        json.dump(datas_final,f,indent = 4)
    print('total datas_final len : ',len(datas_final))
if __name__ == '__main__':
    main()