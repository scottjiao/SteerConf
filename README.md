
# Release code of SteerConf


 

## How to run SteerConf 

Put your API key in `api_key.txt`, and change the "base_site" in `llm_query_helper.py` to the base site of your LLM provider.


## Query in different steering directions and levels (take Llama-3.3-70b-instruct with CoT as an example)

python scripts/20241209_confidence_after_answer_important_decisions_llama.py 

## Aggregate results

python scripts/20241209_result_aggregate.py --cot "cot" --model Llama-3.3-70b-instruct --datasets_list "0123456" --dates "02-02-14-49"

## Visualization and evaluation

bash scripts/20241209_confidence_after_answer_important_decisions_llama_summarized.sh


