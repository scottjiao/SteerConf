 
PROMPT_TYPE="confidence_after_answer_important_decisions_summarized"
SAMPLING_TYPE="self_random" 
NUM_ENSEMBLE=1
CONFIDENCE_TYPE="${PROMPT_TYPE}_${SAMPLING_TYPE}_${NUM_ENSEMBLE}"
USE_COT=true



TIME_STAMPE="01-08-16-05"


#model="Llama-3.3-70b-instruct"
#cot=""
#cot="cot"
#dates="01-08-16-05"

MODEL_NAME="Llama-3.3-70b-instruct"
TEMPERATURE=0.7



dns=( "BigBench_sportUND" "Business_Ethics"  "Professional_Law"  "BigBench_ObjectCounting"  "BigBench_strategyQA"  "BigBench_DateUnderstanding" "GSM8K")
tts=("multi_choice_qa"  "multi_choice_qa"  "multi_choice_qa"  "open_number_qa"  "multi_choice_qa"  "multi_choice_qa" "open_number_qa" )
dps=("dataset/BigBench/sport_understanding.json"  "dataset/MMLU/business_ethics_test.csv"  "dataset/MMLU/professional_law_test.csv"  "dataset/BigBench/object_counting.json"  "dataset/BigBench/strategy_qa.json"  "dataset/BigBench/date_understanding.json" "dataset/grade_school_math/data/test.jsonl")

#dns=(  "BigBench_strategyQA" )
#tts=( "multi_choice_qa"  )
#dps=( "dataset/BigBench/strategy_qa.json" )


# iterate over different settings
for ((i=0;i<${#dns[@]};++i));
do
    DATASET_NAME=${dns[i]}
    TASK_TYPE=${tts[i]}
    DATASET_PATH=${dps[i]}
    echo $DATASET_NAME
    echo $TASK_TYPE
    echo $DATASET_PATH



    #############################################################
    # set time stamp to differentiate the output file

    USE_COT_FLAG=""
    USE_COT_Dir=""

    if [ "$USE_COT" = true ] ; then
        USE_COT_FLAG="--use_cot"
        USE_COT_Dir="cot"
    fi

    OUTPUT_DIR="final_output/${CONFIDENCE_TYPE}_${USE_COT_Dir}/${MODEL_NAME}/${DATASET_NAME}"
    RESULT_FILE="$OUTPUT_DIR/${DATASET_NAME}_${MODEL_NAME}_${TIME_STAMPE}.json"

    echo python query_vanilla_or_cot.py \
    --dataset_name  $DATASET_NAME \
    --data_path $DATASET_PATH \
    --output_file  $RESULT_FILE \
    --model_name  $MODEL_NAME \
    --task_type  $TASK_TYPE  \
    --prompt_type $PROMPT_TYPE \
    --sampling_type $SAMPLING_TYPE \
    --num_ensemble $NUM_ENSEMBLE \
    --temperature_for_ensemble $TEMPERATURE \
    $USE_COT_FLAG


    # uncomment following lines to run test and visualization
    echo python extract_answers.py \
    --input_file $RESULT_FILE \
    --model_name  $MODEL_NAME \
    --dataset_name  $DATASET_NAME \
    --task_type  $TASK_TYPE   \
    --prompt_type $PROMPT_TYPE \
    --sampling_type $SAMPLING_TYPE \
    --num_ensemble $NUM_ENSEMBLE \
        $USE_COT_FLAG

    RESULT_FILE_PROCESSED=$(echo $RESULT_FILE | sed 's/\.json$/_processed.json/')

    #multiply_factors softmax_conf majority_voting reverse_conf_variability
    python vis_aggregated_conf.py \
        --input_file $RESULT_FILE_PROCESSED \
        --model_name  $MODEL_NAME \
        --dataset_name  $DATASET_NAME \
        --task_type  $TASK_TYPE   \
        --prompt_type $PROMPT_TYPE  \
        --sampling_type $SAMPLING_TYPE \
        --num_ensemble $NUM_ENSEMBLE \
        $USE_COT_FLAG    
        #--multiply_factors False \
        #--softmax_conf True \
        #--majority_voting_with_conf false \
        #--reverse_conf_variability False \
        #--poly_linear_conf True \
done