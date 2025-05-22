import json
import os
import sys
import copy
import argparse
currentcwd=os.getcwd() 
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append( os.path.dirname( os.path.dirname(__file__) ) )
sys.path.append( os.path.dirname(__file__) )







argparser = argparse.ArgumentParser(description='Aggregate the results of the model')

argparser.add_argument('--cot', type=str, default="", help='cot')
argparser.add_argument('--dates', type=str, default="01-08-16-05", help='dates')
argparser.add_argument('--model', type=str, default="Llama-3.3-70b-instruct", help='model')
argparser.add_argument('--datasets_list', type=str, default="0123456", help='datasets')



args = argparser.parse_args()







ds_mapping={
    "0":"BigBench_sportUND",
    "1":"Business_Ethics",
    "2":"Professional_Law",
    "3":"BigBench_ObjectCounting",
    "4":"BigBench_strategyQA",
    "5":"BigBench_DateUnderstanding",
    "6":"GSM8K"
}


ds=[ ds_mapping[d] for d in args.datasets_list]

model= args.model


cot=  args.cot 
dates=  args.dates
 

def find_file_end_with(d, end):
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(end):
                return os.path.join(root, file)
    return None
 
def get_summarized_json(d):

    print(f"Processing {d}",end="\t")
    levels_mapping={
        "very_cautious":0,
        "cautious":1,
        "vanilla":2,
        "confident":3,
        "very_confident":4
    }
    input_files={}
    for lv in levels_mapping:
        file=f"final_output/confidence_after_answer_important_decisions_{lv}_self_random_1_{cot}/{model}/{d}/" if lv!="vanilla" else f"final_output/vanilla_self_random_1_{cot}/{model}/{d}/"
        file=find_file_end_with(file, "_processed.json")
        input_files[levels_mapping[lv]]=file
        

    error_counter={} 



    output_file =  f"final_output/confidence_after_answer_important_decisions_summarized_self_random_1_{cot}/{model}/{d}/{d}_{model}_{dates}_processed.json"



    output_json={}
    input_jsons=[]
    failed_ids=[]

    for i in input_files:
        if input_files[i] is None:
            print(f"Missing {d} {i}")
        with open(input_files[i], "r") as f:
            input_json = json.load(f)
            input_jsons.append(input_json)
            failed_ids.append([])
            for key in input_json:
                if key =="hyperparameters":
                    # copy hyperparameters
                    hp=copy.deepcopy(input_json[key])
                    hp["input_file"]=output_file
                    hp["prompt_type"]="confidence_after_answer_important_decisions_summarized"
                    output_json[key]=hp
                elif key in ["sample_tested", "error_count"]:
                    output_json[key]=input_json[key]
                elif key=="processed_data":
                    if "processed_data" not in output_json:
                        output_json["processed_data"]={}
                    for question in input_json[key]:
                        if question not in output_json["processed_data"]:
                            output_json["processed_data"][question]={
                                
                                            }
                        for entry in input_json[key][question]:
                            if entry in ["real_answer","predicted_answer","predicted_conf"]:
                                output_json["processed_data"][question][entry]=input_json[key][question][entry]
                            else:
                                if entry not in output_json["processed_data"][question]:
                                    output_json["processed_data"][question][entry]={}
                                output_json["processed_data"][question][entry][f"trail_{i}"]=input_json[key][question][entry]["trail_0"]
                                

    # condense the trail, there may be trail_0, trail_3, trail_4, we need to make them into trail_0, trail_1, trail_2
    ratios={}
    for question in output_json["processed_data"]:
        for entry in output_json["processed_data"][question]:
            if entry in ["real_answer","predicted_answer","predicted_conf"]:
                continue
            new_entry={}
            """i=0
            while f"trail_{i}" in output_json["processed_data"][question][entry]:
                new_entry[f"trail_{i}"]=output_json["processed_data"][question][entry][f"trail_{i}"]
                i+=1"""
            ks= output_json["processed_data"][question][entry].keys()
            #sort by last
            ks=sorted(ks, key=lambda x: int(x.split("_")[1]))
            for i in range((len(ks))):
                k=ks[i]
                new_entry[f"trail_{i}"]=output_json["processed_data"][question][entry][k]
            if len(ks)!=5 and entry=="hint_confs":
                for i in range(len(input_jsons)):
                    if question in input_jsons[i]["processed_data"]:
                        prmt=input_jsons[i]["processed_data"][question][entry]
                    else:
                        if i not in error_counter:
                            error_counter[i]=[]
                        error_counter[i].append(question)
                    


            output_json["processed_data"][question][entry]=new_entry
            if len(ks) not in ratios:
                ratios[len(ks)]=0
            if entry=="hint_confs":
                ratios[len(ks)]+=1
    print(ratios)
    for i in error_counter:
        print(i,len(error_counter[i]))

    # recurssively mkdir
    def mkdir(path):
        if not os.path.exists(path):
            mkdir(os.path.dirname(path))
            os.mkdir(path)
        return path

    mkdir(os.path.dirname(output_file))


    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=2)



for d in ds:
    get_summarized_json(d)



