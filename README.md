
# Release code of SteerConf: Steering LLMs for Confidence Elicitation
Ziang Zhou, Tianyuan Jin, Jieming Shi, Qing Li (https://arxiv.org/pdf/2503.02863)

*2025/09/19* Our work SteerConf has been accepted by NeurIPS 25!
 

## How to run SteerConf 

Put your API key in `api_key.txt`, and change the "base_site" in `llm_query_helper.py` to the base site of your LLM provider.


## Query in different steering directions and levels (take Llama-3.3-70b-instruct with CoT as an example)

python scripts/confidence_after_answer_important_decisions_llama.py 

## Aggregate results

python scripts/result_aggregate.py --cot "cot" --model Llama-3.3-70b-instruct --datasets_list "0123456"  

## Visualization and evaluation

bash scripts/confidence_after_answer_important_decisions_llama_summarized.sh


