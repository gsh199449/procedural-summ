from typing import Dict, Union, Any, Optional, List, Tuple

import torch
from torch import nn
from transformers.trainer_seq2seq import Seq2SeqTrainer
from torch.utils.data.dataloader import DataLoader


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        把输入挪到正确的device上
        """
        super(MySeq2SeqTrainer, self)._prepare_inputs(inputs)
        # for sample in inputs['sentence_entity']:
        #     for sent in sample:
        #         for key in sent:  # key in 'entity' 'after' 'before'
        #             sent[key]['input_ids'] = torch.tensor(sent[key]['input_ids']).to(self.args.device)
        #             sent[key]['attention_mask'] = torch.tensor(sent[key]['attention_mask']).to(self.args.device)
        for sample in inputs['all_state']:
            sample['input_ids'] = torch.tensor(sample['input_ids']).to(self.args.device)
            sample['attention_mask'] = torch.tensor(sample['attention_mask']).to(self.args.device)
        for sample in inputs['transition_entity']:
            sample['input_ids'] = torch.tensor(sample['input_ids']).to(self.args.device)
            sample['attention_mask'] = torch.tensor(sample['attention_mask']).to(self.args.device)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "source": inputs['source'],
            "target": inputs['target'],
            'after_state_id': inputs['after_state_id'],
            'before_state_id': inputs['before_state_id'],
            'all_state': inputs['all_state'],
            'transition_entity': inputs['transition_entity'],
            # "addi_source_attention_mask": inputs['addi_source_attention_mask'],
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def num_examples(self, dataloader: DataLoader) -> int:
        return sum([len(d) for d in dataloader.dataset.data['summary']])