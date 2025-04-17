from typing import List
from datasets import load_dataset, Dataset
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "/home/qingtaoli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
MAX_TOKENS = 16384
START_NUM = 0
SELECT_NUM = 93733
SAVE_PATH = "/home/qingtaoli/data/teacher_gen/Qwen-14B_with_verify"

# MIN_TOKENS = 1000
MIN_TOKENS = 0
BATCH_SIZE = 4000
CHECKPOINT_FILE = os.path.join(SAVE_PATH, "checkpoint.txt")

os.makedirs(SAVE_PATH, exist_ok=True)


class EvalSample:
    def __init__(self, sample, i, gold_answer, generated_answer, answer_to_verify):
        self.sample = sample
        self.i = i
        self.gold_answer = gold_answer
        self.generated_answer = generated_answer
        self.answer_to_verify = answer_to_verify



print("加载原始数据集...")
dataset = load_dataset(DATASET_NAME, name="default", split="train")
# sub_dataset = dataset.shuffle(seed=42).select(range(SELECT_NUM))
sub_dataset = dataset.select(range(START_NUM, START_NUM + SELECT_NUM, 1))
dsl = sub_dataset.to_list()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
for index, s in tqdm(enumerate(dsl)):
    messages = [
            {"role": "user", "content": s['problem'] + "\nPlease reason step by step, and put your final answer within \\boxed{}."},
        ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    s['index'] = index
    s['prompted_question'] = text

print("\n问题-答案示例：")
print(dsl[8]['prompted_question'])
print(dsl[8]['answer'])
print()


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


### Start generation ###
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
    if '14B_generations' not in s:
        s['14B_generations'] = ["<think>\n" + this_answer]
        s['correctness_answer_verify'] = [is_correct]
        s['correctness_answer_verify_model'] = [is_correct_model]
        s['verify_model_vote'] = [eval_vote]
    else:
        s['14B_generations'].append("<think>\n" + this_answer)
        s['correctness_answer_verify'].append(is_correct)
        s['correctness_answer_verify_model'].append(is_correct_model)
        s['verify_model_vote'].append(eval_vote)


selected_indices = []
selected_questions = []
original_answers = []
generations = []
correctness_answer_verify = []
correctness_answer_verify_model = []
verify_model_vote = []

total_batches = (len(dsl) + BATCH_SIZE - 1) // BATCH_SIZE
start_batch = 0
print(f"开始批量处理（共 {total_batches - start_batch} 个批次）...")
for batch_idx in tqdm(range(start_batch, total_batches)):
    
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx + 1) * BATCH_SIZE
    
    current_samples = dsl[start_idx:end_idx]
    current_questions = [s['prompted_question'] for s in current_samples]
    for attempt_i in range(4):
        outputs = llm.generate(current_questions, sampling_params)
        
        next_round_indices = []
        eval_samples : List[EvalSample] = []
        for i, (s, output) in tqdm(enumerate(zip(current_samples, outputs))):
            # assert i == s['index'] - start_idx - 1
            if not output.outputs:
                print(f"第 {i + 1} 个样本生成失败，跳过\n")
                record_vefify_results_to_dataset_sample(s, "", None, None, 0)
                next_round_indices.append(i)
                continue
            
            generated_output = output.outputs[0]
            token_count = len(generated_output.token_ids)
            if token_count < MIN_TOKENS or MAX_TOKENS - 100 <= token_count:
                print(f"第 {i + 1} 个样本生成长度过长，跳过\n")
                record_vefify_results_to_dataset_sample(s, "", None, None, 0)
                next_round_indices.append(i)
                continue

            gold_answer = s['answer'].replace("\\\\", "\\")
            this_answer = generated_output.text.strip()
            # this_answer = this_answer.replace("\\dfrac", "\\frac")

            answer_to_verify = this_answer.split("</think>")[-1].strip()
            boxed_contents = extract_boxed(answer_to_verify)
            answer_to_verify = ", ".join(boxed_contents)
            gold = parse(gold_answer, extraction_config=[LatexExtractionConfig(boxed_match_priority=-1), ExprExtractionConfig(), StringExtractionConfig()])
            iron = parse(answer_to_verify, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()])
            is_correct = verify(gold, iron)

            if not is_correct:
                eval_prompt = f"I will give you a question, a gold answer and a generated answer, please tell me whether the generated answer is correct. There's no need to check the reasoning process, you should be brief and only focus on the final answer. \n\n<Question>: {s['problem']}\n\n<Gold Answer>: {gold_answer}\n\n<Generated Answer>: {answer_to_verify}\n\n Remember that you should be brief! Only give one word as output: TRUE or FALSE. "
                eval_samples.append(EvalSample(s, i, gold_answer, this_answer, answer_to_verify))
                continue
            else:
                print(f"\n判断 {is_correct}。第 {i + 1} 个样本的答案：{answer_to_verify} <ext> {iron}。原答案：{gold_answer} <ext> {gold}。")
                print(f"原始输出：{this_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
                record_vefify_results_to_dataset_sample(s, "<think>\n" + this_answer, is_correct, is_correct_model=None, eval_vote=0)

        if len(eval_samples) > 0:
            # Gather all the prompts to make it more efficient
            num_judges = 3
            eval_prompts = []
            for eval_sample in eval_samples:
                eval_prompt = f"I will give you a question, a gold answer and a generated answer, please tell me whether the generated answer is correct. There's no need to check the reasoning process, you should be brief and only focus on the final answer. \n\n<Question>: {eval_sample.sample['problem']}\n\n<Gold Answer>: {eval_sample.gold_answer}\n\n<Generated Answer>: {eval_sample.answer_to_verify}\n\n Remember that you should be brief! Only give one word as output: TRUE or FALSE. "
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
                if eval_vote > 0:
                    is_correct_model = True
                else:
                    is_correct_model = False

                print(f"\n判断 False, {is_correct_model}。第 {eval_sample.i + 1} 个样本的答案：{eval_sample.answer_to_verify}。原答案：{eval_sample.gold_answer}。")
                print(f"原始输出：{eval_sample.generated_answer.split('</think>')[-1].strip()}\n------------------------------------------------\n")
                print(f"评委总分：{eval_vote}\n------------------------------------------------\n")
                record_vefify_results_to_dataset_sample(eval_sample.sample, "<think>\n" + eval_sample.generated_answer, False, is_correct_model, eval_vote)

            if not is_correct_model:
                next_round_indices.append(eval_sample.i)
        
        current_questions = [current_questions[i] for i in next_round_indices]
        current_samples = [current_samples[i] for i in next_round_indices]
        if len(next_round_indices) == 0:
            break

    selected_indices.extend([s['index'] for s in dsl[start_idx:end_idx]])
    selected_questions.extend([s['problem'] for s in dsl[start_idx:end_idx]])
    original_answers.extend([s['answer'] for s in dsl[start_idx:end_idx]])
    generations.extend([s['14B_generations'] for s in dsl[start_idx:end_idx]])
    correctness_answer_verify.extend([s['correctness_answer_verify'] for s in dsl[start_idx:end_idx]])
    correctness_answer_verify_model.extend([s['correctness_answer_verify_model'] for s in dsl[start_idx:end_idx]])
    verify_model_vote.extend([s['verify_model_vote'] for s in dsl[start_idx:end_idx]])

    Dataset.from_dict({
            "index": selected_indices,
            "question": selected_questions,
            "original_answer": original_answers,
            "generations": generations,
            "generated_answers": [[gg.split("</think>")[-1].strip() for gg in g] for g in generations],
            "correctness_answer_verify": correctness_answer_verify,
            "correctness_answer_verify_model": correctness_answer_verify_model,
            "verify_model_vote": verify_model_vote,
        }).save_to_disk(f"{SAVE_PATH}/checkpoint_{batch_idx}")
    print(f"第 {batch_idx + 1} 个批次处理完成，已存 {len(selected_questions)} 条有效数据")
    

Dataset.from_dict({
        "index": selected_indices,
        "question": selected_questions,
        "original_answer": original_answers,
        "generations": generations,
        "generated_answers": [[gg.split("</think>")[-1].strip() for gg in g] for g in generations],
        "correctness_answer_verify": correctness_answer_verify,
        "correctness_answer_verify_model": correctness_answer_verify_model,
        "verify_model_vote": verify_model_vote,
    }).save_to_disk(SAVE_PATH)
print(f"处理完成！最终保存 {len(selected_questions)} 条有效数据")

print("全流程执行完毕！")
