import os
import subprocess
from multiprocessing import Pool
import time
 
PROMPT_TYPES = [
    "confidence_after_answer_important_decisions_very_cautious",
    "confidence_after_answer_important_decisions_cautious",
    "vanilla",
    "confidence_after_answer_important_decisions_confident",
    "confidence_after_answer_important_decisions_very_confident", 
]
SAMPLING_TYPE = "self_random"
NUM_ENSEMBLE = 1 
TIME_STAMP = "01-08-16-05"
#TIME_STAMP = time.strftime("%m-%d-%H-%M")
 
USE_COT = True   
MODEL_NAME = "Llama-3.3-70b-instruct"
TEMPERATURE = 0.7
 
NUM_PROCESSES = 1  
 
dns = [
    "BigBench_sportUND", "Business_Ethics", "Professional_Law",
    "BigBench_ObjectCounting", "BigBench_strategyQA", "BigBench_DateUnderstanding", 
    "GSM8K"
]
"""tts = [
    "multi_choice_qa", "multi_choice_qa", "multi_choice_qa",
    "open_number_qa", "multi_choice_qa", "multi_choice_qa", "open_number_qa"
]
dps = [
    "dataset/BigBench/sport_understanding.json", "dataset/MMLU/business_ethics_test.csv",
    "dataset/MMLU/professional_law_test.csv", "dataset/BigBench/object_counting.json",
    "dataset/BigBench/strategy_qa.json", "dataset/BigBench/date_understanding.json",
    "dataset/grade_school_math/data/test.jsonl"
]"""


dn_tt_dp_dict={
    "BigBench_sportUND": ("multi_choice_qa", "dataset/BigBench/sport_understanding.json"),
    "Business_Ethics": ("multi_choice_qa", "dataset/MMLU/business_ethics_test.csv"),
    "Professional_Law": ("multi_choice_qa", "dataset/MMLU/professional_law_test.csv"),
    "BigBench_ObjectCounting": ("open_number_qa", "dataset/BigBench/object_counting.json"),
    "BigBench_strategyQA": ("multi_choice_qa", "dataset/BigBench/strategy_qa.json"),
    "BigBench_DateUnderstanding": ("multi_choice_qa", "dataset/BigBench/date_understanding.json"),
    "GSM8K": ("open_number_qa", "dataset/grade_school_math/data/test.jsonl")
}






def process_all_steps(task):
    i, PROMPT_TYPE = task
    # 计算 CONFIDENCE_TYPE
    CONFIDENCE_TYPE = f"{PROMPT_TYPE}_{SAMPLING_TYPE}_{NUM_ENSEMBLE}" 
    DATASET_NAME = dns[i]
    TASK_TYPE = dn_tt_dp_dict[DATASET_NAME][0]
    DATASET_PATH =  dn_tt_dp_dict[DATASET_NAME][1]
    print(f"Processing dataset: {DATASET_NAME} with PROMPT_TYPE: {PROMPT_TYPE}")
    print(f"Task type: {TASK_TYPE}")
    print(f"Dataset path: {DATASET_PATH}")

    #USE_COT_FLAG = "--use_cot" if USE_COT else ""
 
    cot_flag_dir = 'cot' if USE_COT else ''
    OUTPUT_DIR = os.path.join(
        "final_output",
        f"{CONFIDENCE_TYPE}_{cot_flag_dir}",
        MODEL_NAME,
        DATASET_NAME
    )
    RESULT_FILE = os.path.join(
        OUTPUT_DIR,
        f"{DATASET_NAME}_{MODEL_NAME}_{TIME_STAMP}.json"
    )
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    query_command = [
        "python", "query_vanilla_or_cot.py",
        "--dataset_name", DATASET_NAME,
        "--data_path", DATASET_PATH,
        "--output_file", RESULT_FILE,
        "--model_name", MODEL_NAME,
        "--task_type", TASK_TYPE,
        "--prompt_type", PROMPT_TYPE,
        "--sampling_type", SAMPLING_TYPE,
        "--num_ensemble", str(NUM_ENSEMBLE),
        "--temperature_for_ensemble", str(TEMPERATURE)
    ]
    if USE_COT:
        query_command.append("--use_cot")

    print('Running command:', ' '.join(query_command))
    subprocess.run(query_command)
 
    extract_command = [
        "python", "extract_answers.py",
        "--input_file", RESULT_FILE,
        "--model_name", MODEL_NAME,
        "--dataset_name", DATASET_NAME,
        "--task_type", TASK_TYPE,
        "--prompt_type", PROMPT_TYPE,
        "--sampling_type", SAMPLING_TYPE,
        "--num_ensemble", str(NUM_ENSEMBLE)
    ]
    if USE_COT:
        extract_command.append("--use_cot")

    subprocess.run(extract_command)
 
    RESULT_FILE_PROCESSED = RESULT_FILE.replace(".json", "_processed.json")
 
    vis_command = [
        "python", "vis_aggregated_conf.py",
        "--input_file", RESULT_FILE_PROCESSED,
        "--model_name", MODEL_NAME,
        "--dataset_name", DATASET_NAME,
        "--task_type", TASK_TYPE,
        "--prompt_type", PROMPT_TYPE,
        "--sampling_type", SAMPLING_TYPE,
        "--num_ensemble", str(NUM_ENSEMBLE)
    ]
    if USE_COT:
        vis_command.append("--use_cot")

    subprocess.run(vis_command)

if __name__ == '__main__': 
    tasks = [
        (i, PROMPT_TYPE)
        for PROMPT_TYPE in PROMPT_TYPES
        for i in range(len(dns))
    ]
 
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_all_steps, tasks)