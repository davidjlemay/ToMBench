import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from prompts import *
from tqdm import tqdm
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).half().cuda()

    model_name = args.model_name.split("/")[-1]
    data_path = f"{os.environ.get('SLURM_TMPDIR')}/data"
    files = os.listdir(data_path)

    if args.task != "":
        files = [args.task]
    
    for file in files:
        task = file.split(".")[0]
        with open(f"{data_path}/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
        
        print(file)
        for i, d in tqdm(enumerate(data[:10])):
            for j in range(args.try_times):
                if isinstance(d['选项C'], str):
                    maps, prompt = format_prompt_4(d, args)
                else:
                    maps, prompt = format_prompt_2(d, args)
                
                system_prompt = ""
                if args.language == "zh":
                    if args.cot == False:
                        system_prompt = SystemEvaluatePrompt_zh
                    else:
                        system_prompt = SystemEvaluatePrompt_zh_cot
                else:
                    if args.cot == False:
                        system_prompt = SystemEvaluatePrompt_en
                    else:
                        system_prompt = SystemEvaluatePrompt_en_cot

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                gen_kwargs = {"max_length": 4096, "do_sample": False, "top_k": 1}
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True, return_dict=True)
                inputs = inputs.to(model.device)
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
                out = {}
                out['idx'] = i
                out['number'] = j
                out['answer'] = d['答案\nANSWER']
                out['map'] = maps
                out['data'] = d
                out['output'] = outputs

                with open(f"{os.environ.get('SLURM_TMPDIR')}/results/{task}_{model_name}_results.jsonl", "a+", encoding='utf-8') as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
