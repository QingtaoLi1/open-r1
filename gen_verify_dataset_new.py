from typing import List
from datasets import load_dataset, load_from_disk, Dataset
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "/home/qingtaoli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/"
# DATASET_NAME = "open-r1/OpenR1-Math-220k"
VERIFIED_DIR = "/home/qingtaoli/data/teacher_gen/Qwen-14B_new_run/checkpoint_8/"

MAX_TOKENS = 16384
START_NUM = 0
SELECT_NUM = 93733
SAVE_PATH = "/home/qingtaoli/data/teacher_gen/Qwen-14B_new_run"

MIN_TOKENS = 0
BATCH_SIZE = 8000
CHECKPOINT_FILE = os.path.join(SAVE_PATH, "checkpoint.txt")

os.makedirs(SAVE_PATH, exist_ok=True)


class EvalSample:
    def __init__(self, sample, i, gold_answer, generated_answer, answer_to_verify):
        self.sample = sample
        self.i = i
        self.gold_answer = gold_answer
        self.generated_answer = generated_answer
        self.answer_to_verify = answer_to_verify


print("加载验证过的 94k 数据...")
dataset = load_from_disk(VERIFIED_DIR)
sub_dataset = dataset.select(range(START_NUM, START_NUM + SELECT_NUM, 1))
dsl = sub_dataset.to_list()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
for s_o in tqdm(dsl):
    messages = [
            {"role": "user", "content": s_o['question'] + "\nPlease reason step by step, and put your final answer within \\boxed{}."},
        ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    s_o['prompted_question'] = text

# print(dsl[8]['prompted_question'])
# print(dsl[8]['generations'])


### ###
def extract_boxed(text):
    results = []
    i = 0
    while i < len(text):
        if text.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            brace_level = 1
            start = i
            while i < len(text) and brace_level > 0:
                if text[i] == '{':
                    brace_level += 1
                elif text[i] == '}':
                    brace_level -= 1
                i += 1
            results.append(text[start:i-1])
        else:
            i += 1
    return results

def record_vefify_results_to_dataset_sample(s, this_answer, is_correct, is_correct_model, eval_vote):
    s['generations'].append("<think>\n" + this_answer)
    s['generated_answers'].append(this_answer.split("</think>")[-1].strip())
    s['correctness_answer_verify'].append(is_correct)
    s['correctness_answer_verify_model'].append(is_correct_model)
    s['verify_model_vote'].append(eval_vote)


def load_model():
    print("加载模型...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
        dtype="bfloat16"
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=True
    )
    return llm, sampling_params

llm, sampling_params = load_model()



def rule_based_verify(eval_samples: List[EvalSample], origin_sample, i):
    gold_answer = origin_sample['answer'].replace("\\\\", "\\")
    this_answer = origin_sample['dsl_1_answer'].strip()
    this_answer = this_answer.replace("\\dfrac", "\\frac")

    answer_to_verify = this_answer.split("</think>")[-1].strip()
    boxed_contents = extract_boxed(answer_to_verify)
    answer_to_verify = ", ".join(boxed_contents)
    gold = parse(gold_answer, extraction_config=[LatexExtractionConfig(boxed_match_priority=-1), ExprExtractionConfig(), StringExtractionConfig()])
    iron = parse(answer_to_verify, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()])
    is_correct = verify(gold, iron)

    if not is_correct:
        eval_samples.append(EvalSample(origin_sample, i, gold_answer, this_answer, answer_to_verify))
    else:
        # print(f"\n判断 {is_correct}。第 {i + 1 + 1} 个样本的答案：{answer_to_verify} <ext> {iron}。原答案：{gold_answer} <ext> {gold}。")
        # print(f"原始输出：{this_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
        record_vefify_results_to_dataset_sample(origin_sample, this_answer, is_correct, is_correct_model=None, eval_vote=0)

def model_based_verify(eval_samples):
    # Gather all the prompts to make it more efficient
    num_judges = 3
    eval_prompts = []
    for eval_sample in eval_samples:
        eval_prompt = f"I will give you a question, a gold answer and a generated answer, please tell me whether the generated answer is correct. There's no need to check the reasoning process, you should be brief and only focus on the final answer. \n\n<Question>: {eval_sample.sample['question']}\n\n<Gold Answer>: {eval_sample.gold_answer}\n\n<Generated Answer>: {eval_sample.answer_to_verify}\n\n Remember that you should be brief! Only give one word as output: TRUE or FALSE. "
        eval_prompts.extend([eval_prompt] * num_judges)
    eval_outputs = llm.generate(eval_prompts, sampling_params)

    for i, eval_sample in enumerate(eval_samples):
        eval_vote = 0
        is_correct_model = None
        for j in range(num_judges):
            eval_output = eval_outputs[i * num_judges + j]
            eval_result = eval_output.outputs[0].text.split("</think>")[-1].strip()
            if "TRUE" in eval_result:
                eval_vote += 1
            elif "FALSE" in eval_result:
                eval_vote -= 1
        if eval_vote > 0:
            is_correct_model = True
        else:
            is_correct_model = False

        # print(f"\n判断 False, {is_correct_model}。第 {eval_sample.i + 1} 个样本的答案：{eval_sample.answer_to_verify}。原答案：{eval_sample.gold_answer}。")
        # print(f"原始输出：{eval_sample.generated_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
        # print(f"评委总分：{eval_vote}\n------------------------------------------------\n")
        record_vefify_results_to_dataset_sample(eval_sample.sample, eval_sample.generated_answer, False, is_correct_model, eval_vote)



### Start generation ###
total_batches = (len(dsl) + BATCH_SIZE - 1) // BATCH_SIZE
start_batch = 8
print(f"开始批量处理（共 {total_batches - start_batch} 个批次）...")
for batch_idx in tqdm(range(start_batch, total_batches)):
    
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx + 1) * BATCH_SIZE
    
    current_samples = [ sample for sample in dsl[start_idx:end_idx] if (not sample['correctness_answer_verify'][0]) and (sample['verify_model_vote'][0] <= 1) ]
    current_questions = [s['prompted_question'] for s in current_samples]
    for attempt_i in range(4):
        outputs = llm.generate(current_questions, sampling_params)
        
        next_round_indices = []
        eval_samples : List[EvalSample] = []
        for i, (s, output) in tqdm(enumerate(zip(current_samples, outputs))):
            if not output.outputs:
                print(f"第 {s['index'] + 1} 个样本生成失败，跳过\n")
                record_vefify_results_to_dataset_sample(s, "", None, None, 0)
                next_round_indices.append(s['index'])
                continue
            
            generated_output = output.outputs[0]
            token_count = len(generated_output.token_ids)
            if token_count < MIN_TOKENS or MAX_TOKENS - 100 <= token_count:
                print(f"第 {s['index'] + 1} 个样本生成长度过长，跳过\n")
                record_vefify_results_to_dataset_sample(s, "", None, None, 0)
                next_round_indices.append(i)
                continue

            gold_answer = s['original_answer'].replace("\\\\", "\\")
            this_answer = generated_output.text.strip()
            this_answer = this_answer.replace("\\dfrac", "\\frac")

            answer_to_verify = this_answer.split("</think>")[-1].strip()
            boxed_contents = extract_boxed(answer_to_verify)
            answer_to_verify = ", ".join(boxed_contents)
            gold = parse(gold_answer, extraction_config=[LatexExtractionConfig(boxed_match_priority=-1), ExprExtractionConfig(), StringExtractionConfig()])
            iron = parse(answer_to_verify, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()])
            is_correct = verify(gold, iron)

            if not is_correct:
                eval_samples.append(EvalSample(s, i, gold_answer, this_answer, answer_to_verify))
                continue
            else:
                # print(f"\n判断 {is_correct}。第 {s['index'] + 1} 个样本的答案：{answer_to_verify} <ext> {iron}。原答案：{gold_answer} <ext> {gold}。")
                # print(f"原始输出：{this_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
                record_vefify_results_to_dataset_sample(s, this_answer, is_correct, is_correct_model=None, eval_vote=0)

        if len(eval_samples) > 0:
            # Gather all the prompts to make it more efficient
            num_judges = 3
            eval_prompts = []
            for eval_sample in eval_samples:
                eval_prompt = f"I will give you a question, a gold answer and a generated answer, please tell me whether the generated answer is correct. There's no need to check the reasoning process, you should be brief and only focus on the final answer. \n\n<Question>: {eval_sample.sample['question']}\n\n<Gold Answer>: {eval_sample.gold_answer}\n\n<Generated Answer>: {eval_sample.answer_to_verify}\n\n Remember that you should be brief! Only give one word as output: TRUE or FALSE. "
                eval_prompts.extend([eval_prompt] * num_judges)

            eval_outputs = llm.generate(eval_prompts, sampling_params)
            for i, eval_sample in enumerate(eval_samples):
                eval_vote = 0
                is_correct_model = None
                for j in range(3):
                    eval_output = eval_outputs[i * num_judges + j]
                    eval_result = eval_output.outputs[0].text.split("</think>")[-1].strip()
                    if "TRUE" in eval_result:
                        eval_vote += 1
                    elif "FALSE" in eval_result:
                        eval_vote -= 1
                if eval_vote >= 2:
                    is_correct_model = True
                else:
                    is_correct_model = False

                # print(f"\n判断 False, {is_correct_model}。第 {eval_sample.i + 1} 个样本的答案：{eval_sample.answer_to_verify}。原答案：{eval_sample.gold_answer}。")
                # print(f"原始输出：{eval_sample.generated_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
                # print(f"评委总分：{eval_vote}\n------------------------------------------------\n")
                record_vefify_results_to_dataset_sample(eval_sample.sample, eval_sample.generated_answer, False, is_correct_model, eval_vote)

                if not is_correct_model:
                    next_round_indices.append(eval_sample.i)
        
        current_questions = [current_questions[i] for i in next_round_indices]
        current_samples = [current_samples[i] for i in next_round_indices]
        if len(next_round_indices) == 0:
            break

    Dataset.from_list(dsl).save_to_disk(f"{SAVE_PATH}/checkpoint_{batch_idx}")
    print(f"第 {batch_idx + 1} 个批次处理完成，已存 {len(dsl)} 条有效数据")
    

Dataset.from_list(dsl).save_to_disk(SAVE_PATH)
print(f"处理完成！最终保存 {len(dsl)} 条有效数据")

print("全流程执行完毕！")

## Statistics of generated dataset
true, v3, v2, v1, v0, vm1, vm2, vm3, none = 0,0,0,0,0,0,0,0,0
strue, sv3, sv2, sv1, sv0, svm1, svm2, svm3, snone = 0,0,0,0,0,0,0,0,0
for s in dsl:
    gens, vrs, vms, votes = s["generations"], s["correctness_answer_verify"], s["correctness_answer_verify_model"], s["verify_model_vote"]
    for g, r, m, vote in zip(gens, vrs, vms, votes):
        if r:
            true += 1
        elif vote == 3:
            v3 += 1
        elif vote == 2:
            v2 += 1
        elif vote == 1:
            v1 += 1
        elif vote == 0:
            v0 += 1
        elif vote == -1:
            vm1 += 1
        elif vote == -2:
            vm2 += 1
        elif vote == -3:
            vm3 += 1
        else:
            none += 1
    if vrs[-1]:
        strue += 1
    elif votes[-1] == 3:
        sv3 += 1
    elif votes[-1] == 2:
        sv2 += 1
    elif votes[-1] == 1:
        sv1 += 1
    elif votes[-1] == 0:
        sv0 += 1
    elif votes[-1] == -1:
        svm1 += 1
    elif votes[-1] == -2:
        svm2 += 1
    elif votes[-1] == -3:
        svm3 += 1
    else:
        snone += 1

print(f"true: {true}, v3: {v3}, v2: {v2}, v1: {v1}, v0: {v0}, vm1: {vm1}, vm2: {vm2}, vm3: {vm3}, none: {none}")
print(f"strue: {strue}, sv3: {sv3}, sv2: {sv2}, sv1: {sv1}, sv0: {sv0}, svm1: {svm1}, svm2: {svm2}, svm3: {svm3}, snone: {snone}")
