from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os


MODEL_NAME = "/home/qingtaoli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
MAX_TOKENS = 32768
START_NUM = 20000
SELECT_NUM = 20000
SAVE_PATH = "/home/qingtaoli/data/teacher_gen/Qwen-14B/Batch-2"

MIN_TOKENS = 1000
BATCH_SIZE = 4000
CHECKPOINT_FILE = os.path.join(SAVE_PATH, "checkpoint.txt")

os.makedirs(SAVE_PATH, exist_ok=True)

start_batch = 0
selected_questions = []
selected_answers = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


print("加载原始数据集...")
dataset = load_dataset(DATASET_NAME, name="default", split="train")
# sub_dataset = dataset.shuffle(seed=42).select(range(SELECT_NUM))
sub_dataset = dataset.select(range(START_NUM, START_NUM + SELECT_NUM, 1))
prompted_questions = []
for q in sub_dataset["problem"]:
    messages = [
            {"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
        ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    prompted_questions.append(text)

print(prompted_questions[0])



llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=2,
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

total_batches = (len(prompted_questions) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"开始批量处理（共 {total_batches} 个批次）...")
for batch_idx in tqdm(range(start_batch, total_batches)):
    
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx + 1) * BATCH_SIZE
    current_batch = prompted_questions[start_idx:end_idx]
    
    
    outputs = llm.generate(current_batch, sampling_params)
    
    
    batch_questions = []
    batch_answers = []
    for q, output in zip(current_batch, outputs):
        if output.outputs:
            generated_output = output.outputs[0]
            token_count = len(generated_output.token_ids)
            if token_count >= MIN_TOKENS and token_count < MAX_TOKENS:
                batch_questions.append(q)
                batch_answers.append(generated_output.text.strip())
    
    
    selected_questions.extend(batch_questions)
    selected_answers.extend(batch_answers)

    Dataset.from_dict({
            "question": selected_questions,
            "answer": selected_answers
        }).save_to_disk(f"{SAVE_PATH}_checkpoint_{batch_idx}")
    print(f"第 {batch_idx + 1} 个批次处理完成，已存 {len(selected_questions)} 条有效数据")
    



Dataset.from_dict({
        "question": selected_questions,
        "answer": selected_answers
    }).save_to_disk(SAVE_PATH)
print(f"处理完成！最终保存 {len(selected_questions)} 条有效数据")

print("全流程执行完毕！")
