

from collections import Counter
import seaborn as sns
#%%
import json, os, sys, pdb, json
import numpy as np
import os.path as osp
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
from argparse import ArgumentParser
from adjustText import adjust_text
from collections import Counter
import argparse

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'   

#str2bool = lambda x: (str(x).lower() == 'true')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

#%%
################# CONFIG #####################
parser = ArgumentParser()
 

parser.add_argument("--input_file", type=str, default="output/consistency/raw_results_input/BigBench_ObjectCounting_gpt3_2023-04-27_01-09_processed.json")
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--dataset_name", type=str, default="BigBench_DateUnderstanding")
parser.add_argument("--task_type", type=str, default="multi_choice_qa") 

# for prompt strategy
parser.add_argument("--prompt_type", type=str,default="vanilla")  

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, help="number of queries to ensemble for a given question") 
parser.add_argument("--temperature_for_ensemble", type=float, default=0.0) # temperature for ensemble-based methods

 

#multiply_factors softmax_conf majority_voting reverse_conf_variability
parser.add_argument("--multiply_factors", type=str2bool, default=False) # multiply_factors
parser.add_argument("--softmax_conf", type=str2bool, default=True) # softmax_conf
parser.add_argument("--majority_voting_with_conf", type=str2bool, default=False) # majority_voting_with_conf
parser.add_argument("--reverse_conf_variability", type=str2bool, default=False) # reverse_conf_variability
# poly_linear_conf
parser.add_argument("--poly_linear_conf", type=str2bool, default=False) # poly_linear_conf

args = parser.parse_args()

main_folder = os.path.dirname(args.input_file)
input_file_name = os.path.basename(args.input_file)

################## READ DATA ####################

visual_folder = osp.join(main_folder, "visual")
log_folder = osp.join(main_folder, "log")
output_file = osp.join(main_folder, "all_results.csv")
os.makedirs(osp.join(main_folder, "log"), exist_ok=True)
os.makedirs(osp.join(main_folder, "visual"), exist_ok=True)

result_file_error_log = osp.join(log_folder, input_file_name.replace(".json", "_visual_error.log"))
visual_result_file = osp.join(visual_folder, input_file_name.replace(".json", ".png"))

# read all the json files
with open(osp.join(args.input_file), "r") as f:
    data_results = json.load(f)

print("data_results.keys():", data_results.keys())
data = data_results['processed_data']    


# if hyperparmeters are in data, use this to replace args parameters
if 'hyperparameters' in data_results:
    assert args.model_name == data_results['hyperparameters']['model_name']
    assert args.dataset_name == data_results['hyperparameters']['dataset_name']
    assert args.task_type == data_results['hyperparameters']['task_type']
    assert args.use_cot == data_results['hyperparameters']['use_cot'], (args.use_cot, data_results['hyperparameters']['use_cot'])
    assert args.prompt_type == data_results['hyperparameters']['prompt_type'] 
    # sometimes we only use part of the misleading hints to compute the consistency score
    # assert args.num_ensemble <= data_results['hyperparameters']['num_ensembleing_hints']


with open(result_file_error_log, "a") as f:
    print("sample size: ", len(data))
    f.write("sample size: " + str(len(data)) + "\n")
    # print a sample result
    for key, value in data.items():
        print("key:", key)
        for hint_key, hint_value in value.items():
            print(hint_key, ":",  hint_value)
            f.write(str(hint_key) + ":" + str(hint_value) + "\n")
        break


#%%
############### EXTRA INFORMATION FROM RESULTS ####################

if args.dataset_name in ["BigBench_DateUnderstanding"]:
    normal_option_list =  ["A", "B", "C", "D", "E", "F", "G"]
elif args.dataset_name in ["Professional_Law", "Business_Ethics"]:
    normal_option_list = ["A", "B", "C", "D"]
elif args.dataset_name in ["sportUND", "strategyQA", "StrategyQA", "Bigbench_strategyQA", "BigBench_sportUND", "BigBench_strategyQA"]:
    normal_option_list = ["A", "B"]
elif args.dataset_name in ["GSM8K", "BigBench_ObjectCounting"]:
    normal_option_list = None
else:
    raise NotImplementedError(f"Please specify the normal_option_list for this dataset {args.dataset_name}")







def compute_entropy(values):
    count = Counter(values)
    #normal_option_list 
    total = sum(count.values())
    entropy = 0
    for key, value in count.items():
        p = value / total
        if p == 0:
            continue
        entropy -= p * np.log2(p)
    return entropy

def compute_variability(values):
    mean = np.mean(values)
    std_dev = np.std(values)
    if mean == 0:
        return 0.0
    variability = std_dev / mean
    # 将结果归一化到0-1之间
    normalized_variability = variability / (variability + 1)
    return normalized_variability


def compute_shift_summarized_confidence(hint_answers,hint_confs,consistency_score,answer_confs_alltrails):
    # TFTFF as the best
    # if multiply_factors 2==True, then softmax_conf 4 must be False: 24
    #reverse_conf_variability,poly_linear_conf,multiply_factors,majority_voting_with_conf,softmax_conf=summarize_params['reverse_conf_variability'],summarize_params['poly_linear_conf'],summarize_params['multiply_factors'],summarize_params['majority_voting_with_conf'],summarize_params['softmax_conf']
    # TFTFF
    reverse_conf_variability,poly_linear_conf,multiply_factors,majority_voting_with_conf,softmax_conf=True, False, True, args.majority_voting_with_conf, False
    #print("hint_confs:", hint_confs)
    #print("hint_answers:", hint_answers)
    conf_list = [0.01*conf for conf in hint_confs.values()]
    conf_sum = np.sum(conf_list)
    if conf_sum == 0:
        conf_sum=1
        #raise ValueError("conf_sum should not be 0")
    shift_summarized_conf=None
    shift_summarized_answer=None
    # compute the variability of the answers (0-1), 1 means the answers are very different, 0 means the answers are exactly the same
    #variability_answer = compute_entropy(list(hint_answers.values()))
    variability_answer=1-consistency_score
    variability_conf = compute_variability(conf_list)
    mean_conf = np.mean(conf_list)
    # critiria: if variability_answer is small -> the model is confident about the answer
    # if variability_conf is small -> there may be bias/ignorance issue for this question of the model -> should tend to be more conservative

    # what if the variability_conf is high? -> the confidence is very different -> the shift is significant -> the model follows the shifted instruction 

    # if a answer is given with low confidence, this answer is not reliable, thus we could arbitrarily choose the answer with some criteria

    if reverse_conf_variability: #T
        post_variability_conf= 1-variability_conf
    else:
        post_variability_conf=variability_conf 
    sorted_conf_list = sorted(conf_list) 
    if len(sorted_conf_list)==1:
        
        sorted_conf_list.insert(0,0)
        sorted_conf_list.append(1)
    if poly_linear_conf:   
        try:
            a,b,c=np.polyfit(np.linspace(0,1,len(sorted_conf_list)),sorted_conf_list,2)
        except Exception as e: 
            raise e
        conf_index=(1 - variability_answer) *(post_variability_conf)
        shift_summarized_conf=a*(conf_index**2)+b*conf_index+c
         
        #clip
        shift_summarized_conf=max(0,min(1,shift_summarized_conf))
    else:
        #F



    # if multiply_factors==True, then softmax_conf must be False 
        if multiply_factors:  #T
            adjusted_conf = mean_conf * (1 - variability_answer) *(post_variability_conf)
        else:
            assert softmax_conf==True
            adjusted_conf = mean_conf + (1 - variability_answer) + post_variability_conf
 
        if softmax_conf: 

            shift_summarized_conf = 1 / (1 + np.exp(-adjusted_conf))
        else:# F
            shift_summarized_conf = adjusted_conf

    

 
    if majority_voting_with_conf:  
        ans_list = list(hint_answers.values())
        counter = Counter(ans_list)
        # 找出出现频率最高的所有答案
        max_freq = max(counter.values())
        highest_freq_answers = [ans for ans, freq in counter.items() if freq == max_freq]
         
        if len(highest_freq_answers) == 1:
            shift_summarized_answer = highest_freq_answers[0] 
        else: 
            max_conf = -1
            for ans in highest_freq_answers:
                mean_conf = np.mean(answer_confs_alltrails[ans])
                if mean_conf > max_conf:
                    max_conf = mean_conf
                    shift_summarized_answer = ans 
    else:
        # if shift_summarized_conf is low, choose the answer with the lowest confidence, otherwise choose the answer with the highest confidence
        n_bins=len(hint_answers.values())
        min_conf=min(hint_confs.values())
        max_conf=max(hint_confs.values())
        relative_conf=(shift_summarized_conf-min_conf)/(max_conf-min_conf) if max_conf!=min_conf else 0
        idx=int(relative_conf*n_bins)
        if idx<0:
            idx=0
        if idx>=n_bins:
            idx=n_bins-1
        shift_summarized_answer = hint_answers[f"trail_{idx}"]

    
    #print(f"variability_answer: {variability_answer}\tvariability_conf: {variability_conf}\tmean_conf: {mean_conf}\tshift_summarized_conf: {shift_summarized_conf}")

    return shift_summarized_answer,shift_summarized_conf


















def search_vis_answers(result_dict, task_type, prompt_type, sampling_type,args): 

    aggregation_strategy = ['avg_conf', 'avg_multistep_conf', "consistency","no_aggr","summarized"]
    score_dicts = {"real_answer": [],
                   "scores": {}
                   }
    
    for key in aggregation_strategy:
        score_dicts['scores'][key] = {"answer": [], "score": []}

    # for every question in the dataset, get their answers and their corresponding confidences -> can be multiple if num_ensemble > 1
    for key, value in result_dict.items(): 
        real_answer = value['real_answer']
        if sampling_type == "misleading":
            predicted_answer = value['predicted_answer']
            predicted_conf = value['predicted_conf']
         
        hint_answers = value['hint_answers']
        hint_confs = value['hint_confs']
        hint_multi_step_confs = value['hint_multi_step_confs']
        assert len(hint_answers) == len(hint_confs), "(len(hint_answers) should be equivalent to len(hint_confidences))"

        # process into a map: answer -> [conf1, conf2, conf3, ...]
        answer_confs_alltrails = {}
        for trail, ans in hint_answers.items(): 
            if ans is None:
                continue 
            if ans not in answer_confs_alltrails:
                answer_confs_alltrails[ans] = [] 
            conf = hint_confs[trail]  
            answer_confs_alltrails[ans].append(conf)
        
        answer_stepconfs_for_alltrails = {}
        hint_step_confs = {}
        if prompt_type == "multi_step":
            for trail, step_confs in hint_multi_step_confs.items():
                confidence_product = 1
                for step_idx, step_result in step_confs.items():
                    step_confidence = step_result['confidence']
                    confidence_product *= step_confidence
                ans = hint_answers[trail]
                if ans not in answer_stepconfs_for_alltrails:
                    answer_stepconfs_for_alltrails[ans] = []
                answer_stepconfs_for_alltrails[ans].append(confidence_product)
                hint_step_confs[trail] = confidence_product
 

        trail_name=list(hint_answers.items())[0][0]
        score_dicts['scores']['no_aggr']['answer'].append(  hint_answers[trail_name] )
        score_dicts['scores']['no_aggr']['score'].append(  0.01*hint_confs[trail_name] )
             
        # compute consistency
        def compute_consistency_score(hint_answers, sampling_type):
            """every query has a answer, find the most frequent answer and its frequency -> consistency score"""
            top_1_ans = [answer for _, answer in hint_answers.items()]
            counter = Counter(top_1_ans)
            total = len(top_1_ans)
            
            # compute the frequency of each answer
            frequencies = {key: value / total for key, value in counter.items()}
            # find the most frequent answer and its frequency
            if sampling_type == "misleading":
                most_freq_ans = predicted_answer
                print(f"frequencies: {frequencies}, most_freq_ans: {most_freq_ans}, hint_answers: {hint_answers}, value: {value}")
                most_freq_score = frequencies[most_freq_ans]
                return most_freq_ans, most_freq_score
            
            most_freq_ans = max(frequencies, key=frequencies.get)
            most_freq_score = frequencies[most_freq_ans]
            return most_freq_ans, most_freq_score
        
        consistency_answer, consistency_score = compute_consistency_score(hint_answers, sampling_type)
        score_dicts['scores']['consistency']['answer'].append(consistency_answer)
        score_dicts['scores']['consistency']['score'].append(consistency_score)


         
        def compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type):
            conf_list = [conf for conf in hint_confs.values()]
            conf_sum = np.sum(conf_list)
            if conf_sum == 0:
                conf_sum=1
                #raise ValueError("conf_sum should not be 0")
            average_confs = {ans: sum(conf_lists)/conf_sum for ans, conf_lists in answer_confs_alltrails.items()}
            
            if sampling_type == "misleading":
                avg_conf_option = predicted_answer
                avg_confidence = average_confs[avg_conf_option]
                return avg_conf_option, avg_confidence
            
            avg_conf_option = max(average_confs, key=average_confs.get)
            avg_confidence = average_confs[avg_conf_option]
            return avg_conf_option, avg_confidence
        
        avg_conf_option, avg_confidence = compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type)
        
        if prompt_type == "multi_step":
            avg_step_conf_option, avg_step_confidence = compute_avg_confidence(hint_step_confs, answer_stepconfs_for_alltrails, sampling_type)
 
        
        if "summarized" in prompt_type:
            #raise NotImplementedError("summarized prompt type is not implemented yet")
            shift_summarized_answer,shift_summarized_conf = compute_shift_summarized_confidence(hint_answers,hint_confs,consistency_score,answer_confs_alltrails)
            
            score_dicts['scores']['summarized']['answer'].append(shift_summarized_answer)
            score_dicts['scores']['summarized']['score'].append(shift_summarized_conf) 
        
        if task_type == "open_number_qa":
            real_answer = float(real_answer)
            consistency_answer = float(consistency_answer)
            avg_conf_option = float(avg_conf_option)
            if prompt_type == "multi_step":
                avg_step_conf_option = float(avg_step_conf_option)
            
        elif task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]    

            
        score_dicts["real_answer"].append(real_answer)
        score_dicts['scores']['avg_conf']['answer'].append(avg_conf_option)
        score_dicts['scores']['avg_conf']['score'].append(avg_confidence)     
        if prompt_type == "multi_step":
            score_dicts['scores']['avg_multistep_conf']['answer'].append(avg_step_conf_option)
            score_dicts['scores']['avg_multistep_conf']['score'].append(avg_step_confidence)      

        
    print("Total count: ", len(score_dicts['real_answer']))  
    return score_dicts


        
score_dict = search_vis_answers(data, args.task_type, prompt_type=args.prompt_type, sampling_type=args.sampling_type,args=args)    


 

       

#%%
############### DEAL WITH ERRORS ####################
"""
Error type: 

"""
# print(" consistency_scores_by_distance:", consistency_scores_by_distance)


#################### VISUALIZATION FUNCTIONS ####################
def plot_ece_diagram(y_true, y_confs, score_type,add_name=""):
    from netcal.presentation import ReliabilityDiagram
    n_bins = 10
    diagram = ReliabilityDiagram(n_bins)

    plt.figure()
    diagram.plot(np.array(y_confs), np.array(y_true))
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_ece_{score_type}_{add_name}.pdf")), dpi=600)

def plot_confidence_histogram_with_detailed_numbers(y_true, y_confs, score_type, acc, auroc, ece, use_annotation=True,add_name=""):

    plt.figure(figsize=(6, 4))    
    corr_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    # correct_color = plt.cm.tab10(0)
    # wrong_color = plt.cm.tab10(1)
    correct_color = "red" # plt.cm.tab10(0)
    wrong_color = "blue" # plt.cm.tab10(1)
    # plt.bar(range(101), corr_counts, alpha=0.5, label='correct', color='blue')
    # plt.bar(range(101), wrong_counts, alpha=0.5, label='wrong', color='orange')
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.5, label='correct answer', color=correct_color, align='mid', range=(-2.5,102.5))
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.5, label='wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))

    tick_set = []
    annotation_correct_color = "black"
    annotation_wrong_color = "red"
    annotation_texts = []

    for i, count in enumerate(corr_counts):
        if count == 0:
            continue
        if use_annotation:
            annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_correct_color, fontsize=10))
        tick_set.append(i)
            
    for i, count in enumerate(wrong_counts):
        if count == 0:
            continue
        if use_annotation:
            annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_wrong_color, fontsize=10))
        tick_set.append(i)
    adjust_text(annotation_texts, only_move={'text': 'y'})

    if args.use_cot:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name} COT: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"COT: ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16)
    else:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name}: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16)
    
    
    # plt.xlim(50, 100)
    plt.ylim(0, 1.1*max(max(n_correct), max(n_wrong)))
    plt.xticks(tick_set, fontsize=10)
    plt.yticks([])
    plt.xlabel("Confidence (%)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':14})
    plt.tight_layout()
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}_{add_name}.png")), dpi=600)
    #plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.pdf")), dpi=600)

def plot_confidence_histogram(y_true, y_confs, score_type, acc, auroc, ece, use_annotation=True,add_name="",save_data=True):

    plt.figure(figsize=(6, 4))    
    corr_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    # correct_color = plt.cm.tab10(0)
    # wrong_color = plt.cm.tab10(1)
    correct_color =  plt.cm.tab10(0)
    wrong_color = plt.cm.tab10(3)
    # plt.bar(range(101), corr_counts, alpha=0.5, label='correct', color='blue')
    # plt.bar(range(101), wrong_counts, alpha=0.5, label='wrong', color='orange')
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.8, label='Wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.8, label='Correct answer', color=correct_color, align='mid', range=(-2.5,102.5), bottom=np.histogram(wrong_confs, bins=21, range=(-2.5,102.5))[0])

    tick_set = [i*20 for i in range(0,6)]
    annotation_correct_color = "black"
    annotation_wrong_color = "red"
    annotation_texts = []
 
    if args.use_cot:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name} COT: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"COT: ACC {100*acc:.1f} / AUROC {100*auroc:.1f} / ECE {100*ece:.1f}", fontsize=18)
    else:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name}: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"ACC {100*acc:.1f} / AUROC {100*auroc:.1f} / ECE {100*ece:.1f}", fontsize=18)
    
    
    # plt.xlim(47.5, 102.5)
    plt.ylim(0, 1.1*max(n_correct+n_wrong))
    plt.xticks(tick_set, fontsize=24)
    #plt.yticks([])
    # y tick for highest and middle
    plt.yticks([0, max(n_correct+n_wrong)/2, max(n_correct+n_wrong)], fontsize=24)
    plt.xlabel("Confidence (%)", fontsize=32)
    plt.ylabel("Count", fontsize=32)
    plt.legend(loc='upper left', prop={ 'size':20})
    plt.tight_layout()



    if save_data:
        data_dict = {
            "metric_name": score_type,
            "acc": acc,
            "auroc": auroc,
            "ece": ece,
            "corr_confs": corr_confs,
            "wrong_confs": wrong_confs,
            "use_cot": args.use_cot,
            "dataset_name": args.dataset_name,
            "model_name": args.model_name
        }
        data_file_path = osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}_{add_name}_data.json"))
        with open(data_file_path, 'w') as f:
            json.dump(data_dict, f)









    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}_{add_name}.pdf")), dpi=600)
    #plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.pdf")), dpi=600)


#%%
#################### COMPUTE ACC/ECE/AUCROC ####################
result_matrics = {}
from utils.compute_metrics import compute_conf_metrics
real_answers = score_dict["real_answer"]
for metric_name, values in score_dict['scores'].items():
    predicted_answers = values['answer']
    predicted_confs = values['score']
    if len(predicted_answers) != len(real_answers):
        continue
    correct = [real_answers[i]==predicted_answers[i] for i in range(len(real_answers))]
    result_matrics[metric_name] = compute_conf_metrics(correct, predicted_confs)
    
    # auroc visualization
    plot_confidence_histogram(correct, predicted_confs, metric_name, result_matrics[metric_name]['acc'], result_matrics[metric_name]['auroc'], result_matrics[metric_name]['ece'], use_annotation=True)
    plot_ece_diagram(correct, predicted_confs, score_type=metric_name)
 
 

        

 
with open(output_file, "w") as f:
    f.write("dataset_name,model_name,prompt_type,sampling_type,aggregation_type,acc,ece,auroc,auprc_p,auprc_n,use_cot,input_file_name,num_ensemble\n")
    for key, values in result_matrics.items():
        f.write(f"{args.dataset_name},{args.model_name},{args.prompt_type},{args.sampling_type},{key},{values['acc']:.5f},{values['ece']:.5f},{values['auroc']:.5f},{values['auprc_p']:.5f},{values['auprc_n']:.5f},{args.use_cot},{input_file_name},{args.num_ensemble}\n")



