import openai
from data_process import read_gsm8k,read_gsmhard
from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
from vllm import LLM, SamplingParams
from utils import batch_prompt,build_batch_datas,map_output,build_infer2infer_prompt,extract_output



def main():
    file = '/data/lyz/math_dpo/datas/infer2infer/gsm8k_infer2infer_sample.json'
    output_file = '/data/lyz/math_dpo/output/gsm8k_infer2infer_gen.json'
    with open(file) as f:
        datas_select = json.load(f)
    print('total len: ',len(datas_select))

    llm = LLM(model="/data/lyz/qwen/Qwen2-7B-Instruct", trust_remote_code=True, tensor_parallel_size=2,tokenizer_mode="auto", dtype="auto",gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.9,max_tokens=512)
    start_time = time.time()
    datas_final =[]
    gen_num = 10
    for data in tqdm(datas_select):
        combined_steps = data['combined_steps']
        if combined_steps:
            batch = build_infer2infer_prompt(data,gen_num)
            outputs = llm.chat(batch,
                            sampling_params=sampling_params,
                            use_tqdm=False)
            data['infer2infer_gen'] = extract_output(outputs)
            data['gen_num'] = gen_num
            datas_final.append(data)
    end_time = time.time()
    print('total time spend: ',end_time - start_time)
    with open(output_file,'w') as f:
        json.dump(datas_final,f,indent = 4)
    print('total datas_final len : ',len(datas_final))
if __name__ == '__main__':
    main()