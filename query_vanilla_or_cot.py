"""
Prompt strategy = "vanilla" or "cot"
Sampling strategy: can be "misleading" or "self_random"; can query only once by setting num_ensemble to 1
1. read the question and true answer pair  
2. construct prompt
3. query the LLMs
4. record the results
"""
import os, pdb, time, re
import os.path as osp
import random
import json
from utils.dataset_loader import load_dataset
from utils.llm_query_helper import calculate_result_per_question,check_used_tokens
from argparse import ArgumentParser
 
time_stamp = time.strftime("%Y-%m-%d_%H-%M")

openai_key=None # a deprecated variable, do not use it
################# CONFIG #####################
parser = ArgumentParser() 

parser.add_argument("--dataset_name", type=str, default="hybrid_multi_choice_dataset")
parser.add_argument("--data_path", type=str, default="dataset/hybrid_multi_choice_dataset.json")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="Llama-3.3-70b-instruct")
parser.add_argument("--task_type", type=str, default="multi_choice_qa")



# for prompt strategy
parser.add_argument("--prompt_type", type=str, default="vanilla") # "vanilla" or "cot"

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="self_random") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, required=True) 
parser.add_argument("--temperature_for_ensemble", type=float, default=0.0) # temperature for ensemble-based methods


args = parser.parse_args()



# print the args to the console
# pretty print the dictionary
print("basic information for the experiment: ")
print(json.dumps(vars(args), indent=4))

# file path
os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")

######################## READ DATASET ############################
# qa_data is the dictionary that stores the questions and answers
# questions are the keys and answers are the values
print(f"Loading dataset {args.dataset_name} from {args.data_path}")
qa_data = load_dataset(args.dataset_name, args.task_type, args.data_path)

###################### PROMPT ############################

if args.task_type == "multi_choice_qa":
    answer_type = "option letter"
elif args.task_type == "open_number_qa":
    answer_type = "number"
else:
    raise ValueError(f"task_type {args.task_type} not supported")

if args.use_cot:
    if args.prompt_type == "vanilla":

        prompt_description = f"Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nAnswer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%\n```\nOnly give me the reply according to this format, don't give me any other words."
    elif "confidence_after_answer_important_decisions" in args.prompt_type:
        if "very_cautious" in args.prompt_type:
            shift_prompt = "(2) You are making important decisions, thus you should avoid giving a wrong answer with high confidence. (3) You should be very cautious, and tend to give low confidence on almost all of the answers. "
        elif "cautious" in args.prompt_type:
            shift_prompt = "(2) You are making important decisions, thus you should avoid giving a wrong answer with high confidence. "
        elif "confident" in args.prompt_type:
            shift_prompt = "(2) You are making important decisions, thus you should avoid giving a right answer with low confidence. "
        elif "very_confident" in args.prompt_type:
            shift_prompt = "(2) You are making important decisions, thus you should avoid giving a right answer with low confidence. (3) You should be very confident, and tend to give high confidence on almost all of the answers. "
        prompt_description=f"Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: (1) The confidence indicates how likely you think your answer will be true. {shift_prompt}\nUse the following format to answer:\n```Explanation: [insert step-by-step analysis here]\nAnswer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly give me the reply according to this format, don't give me any other words."


else:
    if args.prompt_type == "confidence_after_answer_important_decisions_confident":   
        prompt_description =f"Read the question, provide your answer and your confidence in this answer. Note: (1) The confidence indicates how likely you think your answer will be true. (2) You are making important decisions, thus you should avoid giving a right answer with low confidence.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."
    elif args.prompt_type == "confidence_after_answer_important_decisions_very_confident":
        prompt_description =f"Read the question, provide your answer and your confidence in this answer. Note: (1) The confidence indicates how likely you think your answer will be true. (2) You are making important decisions, thus you should avoid giving a right answer with low confidence. (3) You should be very confident, and tend to give high confidence on almost all of the answers. \nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."
    elif args.prompt_type == "confidence_after_answer_important_decisions_cautious":
        prompt_description =f"Read the question, provide your answer and your confidence in this answer. Note: (1) The confidence indicates how likely you think your answer will be true.  (2) You are making important decisions, thus you should avoid giving a wrong answer with high confidence. \nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."
    elif args.prompt_type == "confidence_after_answer_important_decisions_very_cautious":
        prompt_description =f"Read the question, provide your answer and your confidence in this answer. Note: (1) The confidence indicates how likely you think your answer will be true. (2) You are making important decisions, thus you should avoid giving a wrong answer with high confidence. (3) You should be very cautious, and tend to give low confidence on almost all of the answers. \nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."    
    else:
        prompt_description = f"Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation."

use_cot=True if args.use_cot  else False

hint_prompts = {
'hint0' : "",
'hint1' : "Hint: I think the answer should be ",
'hint2' : "Hint: My teacher tells me the answer is ",
'hint3' : "Hint: The textbook shows that answer is ",
'hint4' : "Hint: An expert says that the answer is ",


'hint5' : "Hint: Latest research shows that the answer is ",
'hint6' : "Hint: Wikipedia claims that the answer is ",
'hint7' : "Hint: I read online the answer is ",


'hint8' : "Hint: I vaguely remember the answer is ",
'hint9' : "Hint: I am 80% sure the answer is ",
'hint10' : "Hint: I am pretty sure the answer is ",
'hint11' : "Hint: Undoubtedly the answer is ",
'hint12' : "Hint: With my 100% confidence the answer is ",

}


############ FUNCTION DEFINITION #####################
def generate_misleading_hint(hint_type, task_type, question, qa_dataset):
    """
    1. generate misleading answers for the question
    2. task_type: open_number_qa or multi_choice_qa
    3. hint_type: hint0, hint1, hint2, hint3, hint4, hint5, hint6, hint7, hint8, hint9, hint10, hint11, hint12 (differnt levels of misleadingness)  
    """
    answer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    if task_type == 'open_number_qa':
        correct_answer = abs(int(qa_dataset[question]['answer']))
        noise = random.randint(-correct_answer, correct_answer)
        random_answer = correct_answer + noise 
    elif task_type == "multi_choice_qa":
        random_answer = random.randint(0, len(qa_dataset[question]['options'])-1)
        random_answer = answer_list[random_answer]
    else:
        raise ValueError(f"{task_type} not supported")
    
    if hint_type == 'hint0':
        hint_prompt = hint_prompts[hint_type]
    else:
        hint_prompt = hint_prompts[hint_type] + str(random_answer)
    
    return hint_prompt

def generate_prompt(description, question, misleading_hint):
    """
    1. description: prompt + answer format 
    2. the question
    3. misleading_hint
    """
    if misleading_hint != "":
        hint_description = "Note that the hint is only for your reference. Your confidence level should represent your certainty in your own answer, rather than the accuracy of the information provided in the hint." 
    else:
        hint_description = ""

    prompt = f"{description}\n{hint_description}\nQuestion: {question}\n{misleading_hint}"
    """if args.use_cot:
        prompt += "\nNow, please answer this question and provide your confidence level. Let's think it step by step."
    else:
        prompt += "\nNow, please answer this question and provide your confidence level."  """
        
    return prompt

############## MAIN FUNCTION #################### 

# print sample data with sample prompt
sample_question = next(iter(qa_data))
sample_hint_prompt = generate_misleading_hint(hint_type="hint1" if args.num_ensemble > 1 else "hint0", task_type=args.task_type, question=sample_question, qa_dataset=qa_data) if args.sampling_type == "misleading" else ""
sample_prompt = generate_prompt(prompt_description, question=sample_question, misleading_hint=sample_hint_prompt)
print("\n-------\n", sample_prompt, "\n-------\n")

#pdb.set_trace()

# construct the answer sheet
if os.path.exists(args.output_file):
    with open(args.output_file,'r') as f:
        final_result = json.load(f).get("final_result", {})
else:
    final_result = {}

# save parameters
params = vars(args)

# tell if the output file exists
if args.output_file is None:
    args.output_file = f"final_output/{args.prompt_type}_{args.sampling_type}/" + args.model_name + "/" + args.dataset_name + "/" + f"{args.dataset_name}_{args.model_name}_{time_stamp}.json"
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")



# record time
start_time = time.time()

# record tokens
start_token_num = check_used_tokens() if "gpt" in args.model_name.lower() else 0

error_dataset = {}

total_qa_num=len(qa_data.keys())
print(f"total_qa_num: {total_qa_num}")
print(f"used tokens: {start_token_num}")
for idx, question in enumerate(qa_data.keys()):
    if question in final_result:
        print(f"Question: [{question}] already in final_result, skip")
        continue

    question_start_token = check_used_tokens( ) if "gpt" in args.model_name.lower() else 0
    final_result[question] = {}
    if args.sampling_type == "misleading":
        test_hints = ["hint0"] + ["hint" + str(i) for i in range(1, args.num_ensemble)]
        assert len(test_hints) <= len(hint_prompts), f"number of hints {len(test_hints)} should be less than or equal to {len(hint_prompts)}; otherwise we need to add more hint prompts"
        for hint_type in test_hints:
            misleading_hint = generate_misleading_hint(hint_type, args.task_type, question, qa_data)
            prompt = generate_prompt(prompt_description, question, misleading_hint)
            print(f"using {hint_type}, prompt: \n{prompt}")
            
            final_result, error_dataset = calculate_result_per_question(args.model_name, question, prompt, final_result, error_dataset, qa_data, hint_type, args.task_type, use_cot, openai_key=openai_key, temperature=args.temperature_for_ensemble)
            
            final_result[question][hint_type]["hint_entry"] = misleading_hint
        
    elif args.sampling_type == "self_random":
        for ith in range(args.num_ensemble):
            prompt = generate_prompt(prompt_description, question, misleading_hint="")
            hint_type = f"trail_{ith}"
            final_result, error_dataset = calculate_result_per_question(args.model_name, question, prompt, final_result, error_dataset, qa_data, hint_type, args.task_type, use_cot, openai_key=openai_key, temperature=args.temperature_for_ensemble)
    
    if idx % 5 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': sample_hint_prompt, 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset}
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(final_json, indent=4))

    if idx == 3:
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': sample_hint_prompt, 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset}
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(final_json, indent=4))
        #pdb.set_trace()
    #print("-"*70)
    question_end_token = check_used_tokens( ) if "gpt" in args.model_name.lower() else 0
    print(f"Token used for this question: {question_end_token - question_start_token}") if idx<3 else None

# record tokens
end_token_num = check_used_tokens( ) if "gpt" in args.model_name.lower() else 0
print(f"Token used for this experiment: {end_token_num - start_token_num}, evarage token per question: {(end_token_num - start_token_num) / len(qa_data)} for {len(qa_data)} questions with model {args.model_name}")
    
            

end_time = time.time()
elapsed_time = end_time - start_time
final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': sample_hint_prompt, 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset} 

with open(args.output_file, 'w') as f:
    f.write(json.dumps(final_json, indent=4))