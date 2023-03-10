pip3 install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install packaging flask transformers==4.3.3 datasets==1.2.1 nltk absl-py rouge_score comet_ml sacrebleu==1.5.1 dgl-cu101 elasticsearch tensorboard -q
pip3 install gpustat
gpustat -cpu
nvidia-smi

python3 run_mybart.py --model_name_or_path bart-base --do_train --do_eval --train_file train.json --validation_file valid.json --test_file test.json --output_dir das --exp_name make_dataset_story_new --max_source_length xxx --max_target_length xxx

python3 run_mybart.py \
--log_root ./log \
--model_name_or_path ./bart_model \
--save_dataset_path ./procedure-new2-dataset \
--remove_unused_columns False \
--exp_name $1 \
--do_train \
--do_eval \
--eval_steps 1000 \
--evaluation_strategy steps \
--predict_with_generate True \
--output_dir model/ \
--save_steps 1000 \
--save_total_limit 5 \
--num_train_epochs 20 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--dataloader_num_workers 16 \
--use_kl_loss True \
--concat_context True \
--concat_entity_graph False \
--use_entity_graph False 
