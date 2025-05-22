import openai, pdb
import requests
import time


base_site=""

with open("api_key.txt") as f:
    apiKey=f.read().strip()

def check_used_tokens(): 
    header={"Api-key": apiKey}
    response = requests.get(base_site + "/openai/used-tokens", headers=header)
    jsonObj = response.json()
    used_tokens=int(jsonObj["UsedTokens"]) 
    return used_tokens

def get_response(deployName, body,temp, max_tokens,n=1):
    body.update({"temperature": temp, "max_tokens": max_tokens, "n": n})
    header={"api-key": apiKey,"Content-type": "application/json" }
    response = requests.post(base_site + "/openai/deployments/" + deployName + "/chat/completions",  json=body, headers=header )    
    try:
        jsonObj = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error in json response: {response.text}")
        raise
    
    return jsonObj
 
def calculate_result_per_question(model_name, question, prompt, final_result, error_dataset, qa_dataset, hint_type, task_type, use_cot, openai_key, temperature=0.0):
    """
    - final_result is used to record the result of each question
    - error_dataset is used to record the error message of each question, if the error occurs
    - hint_type is used for record the hint type, e.g. hint0, hint1, hint2, hint3, hint4; if hint is not used, then hint_type is 'hint0'
    - use_cot is used to indicate whether to use cot or not
    
    return:
        final_result: updated final_result
        error_dataset: updated error_dataset
    """
    openai.api_key = openai_key 
    
    # run model api and get response
    max_tokens = 2000 if use_cot else 400
    max_req_count = 20
    req_success = False
    while not req_success and max_req_count > 0:
        try:
            if model_name.lower() == 'gpt3.5': 
                response = get_response("gpt35", {"messages": [ {"role": "user", "content": prompt}]}, temperature, max_tokens)

                orginal_anser = response['choices'][0]['message']['content']  
            elif model_name.lower() == 'gpt4o': 
                response = get_response("gpt4o", {"messages": [ {"role": "user", "content": prompt}]}, temperature, max_tokens)

                orginal_anser = response['choices'][0]['message']['content']  
                
            elif model_name.lower() == 'chatgpt-0301': 
                response = get_response("gpt35", {"messages": [ {"role": "user", "content": prompt}]}, temperature, max_tokens)
        
                orginal_anser = response['choices'][0]['message']['content']  
                
            elif model_name.lower() == 'chatgpt-0613': 
                response = get_response("gpt35", {"messages": [ {"role": "user", "content": prompt}]}, temperature, max_tokens)
                orginal_anser = response['choices'][0]['message']['content'] 
 
            
            elif model_name.lower()  == 'gpt4':
                #response = openai.ChatCompletion.create(model="gpt-4",  messages=[{'role':'user','content':prompt}], temperature=temperature, max_tokens=max_tokens)
                response = get_response("gpt4-fc", {"messages": [ {"role": "user", "content": prompt}]}, temperature, max_tokens)
                orginal_anser = response['choices'][0]['message']['content'] 
                 
                
            elif model_name.lower() in[ 'llama_2_7b_chat_hf',"Llama-3.3-70b-instruct".lower()]:
                #raise ValueError(f"{model_name} not supported")
                #pdb.set_trace()
                from utils.api_llama import LlamaChatCompletion
                model_name = "daryl149/llama-2-7b-chat-hf" if model_name.lower() == 'llama_2_7b_chat_hf' else "Llama-3.3-70b-instruct".lower()
                orginal_anser = LlamaChatCompletion(model_name, prompt, max_tokens=max_tokens,temperature=temperature)
                
                
            else:
                raise ValueError(f"{model_name} not supported")            
            
        
            dict_value = {'hint_response': orginal_anser, 'real_answer':qa_dataset[question]}
            final_result[question][hint_type] = dict_value
             
            req_success = True
        
        except Exception as e:
            print(e)
            print(f"max_req_count: {max_req_count}") 
            #{'statusCode': 429, 'message': 'Rate limit is exceeded. Try again in 2 seconds.'}
            if max_req_count > 0:
                max_req_count -= 1
                time.sleep(5)
            if max_req_count==0:
                if question not in error_dataset:
                    error_dataset[question] = {}
                error_dataset[question][hint_type] = {'error_message': str(e), 'real_answer':qa_dataset[question], 'used_prompt': prompt}
                print(f"fail to get llm answer of question: {question}")

    return final_result, error_dataset
