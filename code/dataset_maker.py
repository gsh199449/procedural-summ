import logging
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from args import DataTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from datasets import load_dataset, DownloadConfig

logger = logging.getLogger(__name__)


class DatasetMaker:
    def __init__(self, dataset_saved_path: str, data_args: DataTrainingArguments,
                 training_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizerBase):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.dataset_saved_path = dataset_saved_path
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        self.max_target_length = self.data_args.max_target_length

    def make_dataset(self):
        logger.info('******* Making Dataset **********')
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file
            extension = self.data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))

        if self.training_args.label_smoothing_factor > 0:
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for model. This will lead to loss being calculated twice and will take up more memory"
            )

        logger.info('saving dataset')
        dataset_saved_path = self.dataset_saved_path
        datasets.save_to_disk(dataset_saved_path)
        logger.info(f'******* Dataset Finish {dataset_saved_path} **********')
        return datasets

    def preprocess(self, examples: Dict):
        inputs = []
        for text in examples['text']:
            inputs.extend([sent.replace(' ', '') if self.data_args.chinese_data else sent for sent in text])
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding,
                                      truncation=True, add_special_tokens=False)

        targets = []
        for summ in examples['summary']:
            targets.extend([sent.replace(' ', '') if self.data_args.chinese_data else sent for sent in summ])
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True,
                                    add_special_tokens=False)
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        new_features = []
        for inp_ids, inp_attn, lab in zip(model_inputs['input_ids'], model_inputs['attention_mask'],
                                          labels["input_ids"]):
            new_features.append({
                'attention_mask': inp_attn,
                'input_ids': inp_ids,
                'labels': lab,
            })

        labels = [feature["labels"] for feature in new_features] if "labels" in new_features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in new_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        def tokenize_entity(e):
            return self.tokenizer(e, max_length=3, padding='max_length', truncation=True, add_special_tokens=False)

        entity_list = [tokenize_entity(elist) for elist in examples['entity_list']]
        source = []  # sentences id
        target = []  # entity id
        after_state_id = []
        before_state_id = []

        all_state = []
        all_entity = []
        for batch_idx, sentences_entity in enumerate(examples['sentence_entity']):  # for each article in batch (max iter 1)
            for sent in sentences_entity:  # for each sentence
                for ent_obj in sent:  # for entities in sentence (max iter 3)
                    all_state.append(ent_obj['after'])
                    all_state.append(ent_obj['before'])
                    all_entity.append(ent_obj['entity'])
        all_state = [list(set(all_state))]
        all_entity = [list(set(all_entity))]

        all_state_tokenized = [tokenize_entity(slist if len(slist) > 0 else [self.tokenizer.pad_token]) for slist in
                               all_state]
        for batch_idx, sentences_entity in enumerate(examples['sentence_entity']):  # for each article in batch (max iter 1)
            b_s = []
            b_t = []
            b_asi = []
            b_bsi = []
            for sent_idx, sent in enumerate(sentences_entity):  # for each sentence
                for ent_obj in sent:  # for entities in sentence (max iter 3)
                    b_s.append(sent_idx)
                    b_t.append(all_entity[batch_idx].index(ent_obj['entity']))
                    b_asi.append(all_state[batch_idx].index(ent_obj['after']))
                    b_bsi.append(all_state[batch_idx].index(ent_obj['before']))
            source.append(b_s)
            target.append(b_t)
            after_state_id.append(b_asi)
            before_state_id.append(b_bsi)
        to_return['source'] = source
        to_return['target'] = target
        to_return['after_state_id'] = after_state_id
        to_return['before_state_id'] = before_state_id
        to_return['all_state'] = all_state_tokenized
        to_return['transition_entity'] = entity_list