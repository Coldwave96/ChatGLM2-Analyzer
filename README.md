# ChatGLM2-Analyzer
Finetune ChatGLM2-6B for web attack analize. Detailed finetunr methods please see [here](https://github.com/THUDM/P-tuning-v2).

## Dataset Format
* Single round 
```Json
{
    "question": "http://www.website.com/encode?type=3&id=1%3Cscript%3Ealert(%22xss%22)%3C/script%3E, is this an web attack?",
    "analysis": "Yes, the URL you provided appears to be a possible web attack known as Cross-Site Scripting (XSS) vulnerability. In this case, the URL contains a query parameter named 'id' with a value that seems to be URL-encoded.\n\nThe decoded value of 'id' is: '1<script>alert('xss')</script>'\n\nThis script tag in the URL is indicative of an attempt to inject malicious code into a web page. If a website does not properly sanitize and validate user input, an attacker can exploit this vulnerability to inject harmful scripts into the page, which can lead to unauthorized access, data theft, or manipulation of the website's content.\n\nAs a responsible practice, websites should always validate and sanitize user input, especially when handling data that will be displayed on the website to prevent XSS attacks. Additionally, users should be cautious when clicking on URLs from untrusted sources or websites to avoid potential security risks."
}
```

* Multi rounds
```Json
{
    "question": "question_1st", 
    "analysis": "analysis_1st", 
    "history": []
}
{
    "question": "question_2nd", 
    "analysis": "analysis_2nd", 
    "history": ["question_1st", "analysis_1st"]
}
{
    "question": "question_3rd", 
    "analysis": "analysis_3rd", 
    "history": [
        ["question_1st", "analysis_1st"], 
        ["question_2nd", "analysis_2nd"]
    ]
}
```

## Training Options
* P-Tuning v2: Freeze model weights and then train an adapter.
```Bash
# Update params with `<>` symbols
torchrun --standalone --nnodes=1 --nproc-per-node=<NUM_GPUS> finetune.py \
    --do_train \
    --train_file <train.json> \
    --validation_file <dev.json> \
    --preprocessing_num_workers 10 \
    --prompt_column question \
    --response_column analysis \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir <output_dir> \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate <LR> \
    --pre_seq_len <PRE_SEQ_LEN> \
    --quantization_bit 8

# Train an adapter which support multi-rounds interaction
torchrun --standalone --nnodes=1 --nproc-per-node=<NUM_GPUS> finetune.py \
    --do_train \
    --train_file <train.json> \
    --validation_file <dev.json> \
    --preprocessing_num_workers 10 \
    --prompt_column question \
    --response_column analysis \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir <output_dir> \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate <LR> \
    --pre_seq_len <PRE_SEQ_LEN> \
    --quantization_bit 8
```

* Finetune all the params
```Bash
# Update params with `<>` symbols
deepspeed --num_gpus=4 --master_port <MASTER_PORT> finetune.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file <train.json> \
    --test_file <dev.json> \
    --prompt_column question \
    --response_column analysis \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir <output_dir> \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate <LR> \
    --fp16
```
