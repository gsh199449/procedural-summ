#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for sequence to sequence.
"""
import logging
import sys
import comet
from dataset_maker import DatasetMaker
import traceback
import aistudio_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import math
import os
from trainer import MySeq2SeqTrainer

import nltk  # Here to have a nice missing dependency error message early on
from datasets import DatasetDict

import transformers
from filelock import FileLock
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed, BartConfig
)
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bart.tokenization_bart import BartTokenizer
from gpu_help import get_available_gpu
from compute_metric import MetricCompute
from args import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    from magic_bart import MyBart, MyCometCallback, AutoDecodeCallback, MyDataCollatorForSeq2Seq, MyBartConfig
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))  # type: ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # type: ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments

    if aistudio_utils.is_km():
        aistudio_utils.send_log(f'exp name {data_args.proj_name} {data_args.exp_name}')
        aistudio_utils.send_log(f'app name {aistudio_utils.get_app_name()}')
    training_args.logging_steps = 10
    data_args.log_root = os.path.join(data_args.log_root, data_args.proj_name, data_args.exp_name)
    training_args.output_dir = os.path.join(data_args.log_root, 'model')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        logger.info('checking last_checkpoint')
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info(f'last_checkpoint : {last_checkpoint}')
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # logging.warning(f'last train is ended, continue train on {training_args.output_dir}')
            # last_checkpoint = training_args.output_dir
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Dataset parameters %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not training_args.do_train and (training_args.do_eval or training_args.do_predict) and model_args.model_name_or_path is None:
        # 纯测试且没指定ckpt 就用最新的ckpt
        model_args.model_name_or_path = last_checkpoint if last_checkpoint is not None else get_last_checkpoint(training_args.output_dir)
    if training_args.do_train and last_checkpoint is not None:
        logger.warning(f'using previous checkpoint {last_checkpoint}')
        model_args.model_name_or_path = last_checkpoint

    if model_args.model_name_or_path is None:
        logger.info('******* Initializing model form scratch **********')
        if data_args.chinese_data:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        else:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.add_special_tokens({
            'bos_token': '<s>',
            'eos_token': '</s>',
        })
        config = MyBartConfig(encoder_layers=6, decoder_layers=6, encoder_ffn_dim=2048, decoder_ffn_dim=2048, encoder_attention_heads=8, decoder_attention_heads=8)
        config.decoder_start_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.num_beams = data_args.num_beams
        config.max_length = data_args.max_target_length
        model = MyBart(config)
    else:
        logger.info(f'******* Loading model form pretrained {model_args.model_name_or_path} **********')
        try:
            tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path)  # 自己训的bart用这行
            logger.info('load BertTokenizerFast')
        except Exception as e:
            logger.error(f'{e} change to load by BartTokenizer')
            tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)  # 如果用bart-base就用这行
            logger.info('load BartTokenizer')
        model = MyBart.from_pretrained(model_args.model_name_or_path)
        logger.info('load model')
    model.config.use_entity_graph = model_args.use_entity_graph
    model.config.concat_entity_graph = model_args.concat_entity_graph
    model.config.use_kl_loss = model_args.use_kl_loss
    if aistudio_utils.is_km():
        aistudio_utils.send_log('\n'.join([f'{n} - {str(getattr(model.config, n))}' for n in ['use_entity_graph', 'concat_entity_graph', 'use_kl_loss']]))
    logger.info('\n'.join([f'{n} - {str(getattr(model.config, n))}' for n in ['use_entity_graph', 'concat_entity_graph', 'use_kl_loss']]))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.save_dataset_path is None:
        maker = DatasetMaker('procedure-story-new-state-dataset', data_args, training_args, tokenizer)
        datasets = maker.make_dataset()
    else:
        logger.info(f'******* Loading Dataset from {data_args.save_dataset_path} **********')
        datasets = DatasetDict.load_from_disk(data_args.save_dataset_path)

    train_dataset = datasets["train"] if training_args.do_train is not None and "train" in datasets else None
    eval_dataset = datasets["validation"] if training_args.do_eval is not None and "validation" in datasets else None
    test_dataset = datasets["test"] if training_args.do_predict is not None and "test" in datasets else datasets["validation"]
    if training_args.do_predict is None and "test" not in datasets:
        logging.warning(f'using validation dataset as test!')

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    max_target_length = data_args.val_max_target_length
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = MyDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            data_args=data_args,
            model_args=model_args
        )

    comp_metric = MetricCompute(data_args, tokenizer, test_dataset, eval_dataset)
    if aistudio_utils.is_km():
        record_file = open(os.path.join(data_args.log_root, f'{aistudio_utils.get_record_id()}.record'), 'w')
        record_file.close()
    # comet_callback = MyCometCallback(data_args.proj_name, data_args.exp_name)

    model.config.num_beams = data_args.num_beams
    model.config.max_length = data_args.max_target_length

    # for arg_class in [model_args, data_args, training_args, model.config]:
    #     for k, v in arg_class.to_dict().items():
    #         comet_callback.exp.experiment.log_parameter(k, v)

    # Initialize our Trainer
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_metric.compute_metrics if training_args.predict_with_generate else None,
        callbacks=[]  # auto_decode_callback comet_callback
    )
    comp_metric.trainer = trainer
    # comet_callback.set_trainer(trainer)

    # Training
    if training_args.do_train:
        try:
            if last_checkpoint is not None:  # 如果是继续之前的训练需要加载步数和optimizer
                train_result = trainer.train(resume_from_checkpoint=model_args.model_name_or_path) # resume_from_checkpoint=checkpoint
            else:
                train_result = trainer.train()
        except KeyboardInterrupt:
            logger.info('stop training')
        finally:
            traceback.print_exc()
            logger.info('exit, saving model')
            trainer.save_model(output_dir=os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}'))  # Saves the tokenizer too for easy upload
            trainer.state.save_to_json(os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}', 'trainer_state.json'))
            exit(0)
        trainer.save_model()

        if trainer.is_world_process_zero():
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        if trainer.state.global_step == 0:
            trainer.state = trainer.state.load_from_json(os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        logger.info(f"*** Evaluate step {trainer.state.global_step} ***")
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        # comet_callback.exp.experiment.log_metric(name="eval_loss", value=eval_output["eval_loss"], step=trainer.state.global_step,
        #                                          epoch=trainer.state.epoch)
        # comet_callback.exp.experiment.log_metric(name="eval_ppl", value=perplexity,
        #                                          step=trainer.state.global_step,
        #                                          epoch=trainer.state.epoch)
        results["perplexity"] = perplexity

        if trainer.is_world_process_zero():
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")

    # predict
    if training_args.do_predict:
        logger.info(f"*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        print(test_results.metrics)
        # for k, v in test_results.metrics.items():
        #     comet_callback.exp.experiment.log_metric(name=k, value=v, step=trainer.state.global_step, epoch=trainer.state.epoch)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_results.label_ids[test_results.label_ids < 0] = tokenizer.pad_token_id
                test_label = tokenizer.batch_decode(
                    test_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                test_labels = [label.strip() for label in test_label]
                for pred, lab in zip(test_preds[:10], test_labels[:10]):
                    logger.info(f'{pred}\t{lab}')

                dec_dir = os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}')
                if not os.path.exists(dec_dir):
                    os.makedirs(dec_dir)
                fo_ref = open(os.path.join(dec_dir, 'reference.txt'), 'w', encoding='utf8')
                fo_dec = open(os.path.join(dec_dir, 'decoded.txt'), 'w', encoding='utf8')
                for pred, lab in zip(test_preds, test_labels):
                    fo_ref.write(f'{lab}\n')
                    fo_dec.write(f'{pred}\n')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    with open('mybart.pid', 'w', encoding='utf8') as w:
        w.write(str(os.getpid()))
    if not aistudio_utils.is_km():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_gpu())
    main()
