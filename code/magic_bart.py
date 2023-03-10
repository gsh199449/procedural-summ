import collections
import logging
import math
import os
import random
import subprocess
import sys
from typing import Optional, Tuple, Union, Dict, Any, List

import dgl
from attr import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import PaddingStrategy

from args import DataTrainingArguments, ModelArguments
from comet import MyExperiment
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from packaging import version

from newhan import HAN

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, \
    ProgressCallback
from transformers.activations import ACT2FN
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, BartAttention,
)
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

logger = logging.getLogger(__name__)


class MyBartConfig(BartConfig):
    def __init__(self, use_entity_graph=True, use_kl_loss=False, concat_entity_graph=True, **kwargs):
        super(MyBartConfig, self).__init__(**kwargs)
        self.use_entity_graph = use_entity_graph
        self.concat_entity_graph = concat_entity_graph
        self.use_kl_loss = use_kl_loss


class MyBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: MyBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([MyBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        graph_hidden=None,
        transition_entity_hidden=None,
        # addi_source_encoder_attention_mask=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # NOT USE expand encoder attention mask
        # if graph_hidden is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     addi_source_encoder_attention_mask = _expand_mask(addi_source_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                    graph_hidden,
                    transition_entity_hidden,
                    # addi_source_encoder_attention_mask,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    graph_hidden=graph_hidden,
                    transition_entity_hidden=transition_entity_hidden,
                    # addi_source_encoder_attention_mask=addi_source_encoder_attention_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MyBartDecoderLayer(nn.Module):
    def __init__(self, config: MyBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        if config.use_entity_graph and not config.concat_entity_graph:
            self.graph_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.graph_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config
        self.rezero = nn.Parameter(torch.ones([1]))
        self.graph_attn_rezero = nn.Parameter(torch.ones([1]))
        self.cross_attn_rezero = nn.Parameter(torch.ones([1]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        graph_hidden=None,
        transition_entity_hidden=None,
        # addi_source_encoder_attention_mask=None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            ###########Concat Graph Hidden############
            if self.config.use_entity_graph and self.config.concat_entity_graph:
                encoder_hidden_states = torch.cat([encoder_hidden_states, graph_hidden.unsqueeze(dim=1)], dim=1)
                batch_size, _, decode_len, encoder_len = encoder_attention_mask.size()
                encoder_attention_mask = torch.cat([encoder_attention_mask, torch.zeros([batch_size, _, decode_len, 1]).to(encoder_attention_mask.device)], dim=3)
            if self.config.use_kl_loss:
                encoder_hidden_states = torch.cat([encoder_hidden_states, transition_entity_hidden.unsqueeze(dim=1)], dim=1)
                batch_size, _, decode_len, encoder_len = encoder_attention_mask.size()
                encoder_attention_mask = torch.cat([encoder_attention_mask, torch.zeros([batch_size, _, decode_len, 1]).to(encoder_attention_mask.device)], dim=3)
            #######################
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states  # self.cross_attn_rezero *
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

            #######################
            if self.config.use_entity_graph and not self.config.concat_entity_graph:
                residual = hidden_states

                # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
                # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.graph_attn(
                    hidden_states=hidden_states,
                    key_value_states=graph_hidden.unsqueeze(dim=0),
                    # past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
                hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
                hidden_states = residual + self.graph_attn_rezero * hidden_states
                hidden_states = self.graph_attn_layer_norm(hidden_states)

                # add cross-attn to positions 3,4 of present_key_value tuple
                # present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + self.rezero * hidden_states
        # hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = MyBartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        graph_hidden=None,
        transition_entity_hidden=None,
        # addi_source_attention_mask=None,
        # addi_source_encoder_outputs=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # addi_source_encoder_outputs = self.encoder(
            #     input_ids=addi_source,
            #     attention_mask=addi_source_attention_mask
            # )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            # addi_source_encoder_outputs = BaseModelOutput(
            #     last_hidden_state=addi_source_encoder_outputs[0],
            #     hidden_states=addi_source_encoder_outputs[1] if len(addi_source_encoder_outputs) > 1 else None,
            #     attentions=addi_source_encoder_outputs[2] if len(addi_source_encoder_outputs) > 2 else None,
            # )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_hidden=graph_hidden,
            transition_entity_hidden=transition_entity_hidden,
            # addi_source_encoder_attention_mask=addi_source_attention_mask,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SimpleAttention2(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, query, values):
        """
        query: [batch, tgt_len, hidden]
        values: [batch, src_len, hidden]
        """
        logits = torch.bmm(query, values.transpose(-1, -2))  # [batch, tgt_len, src_len]
        probs = F.gumbel_softmax(logits, dim=-1, tau=0.8)  # [batch, tgt_len, src_len]
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        attn_output = torch.bmm(probs, values)  # [batch, tgt_len, hidden]
        return attn_output, logits, None


class SimpleAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights = F.softmax(attn_weights, dim=-1)
        attn_logits = attn_weights
        attn_weights = F.gumbel_softmax(attn_weights, dim=-1, tau=0.8)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_logits_reshaped = attn_logits.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
            attn_logits_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, attn_logits_reshaped

class MyBart(BartPretrainedModel):
    config_class = MyBartConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: MyBartConfig):
        super().__init__(config)
        self.model = MyBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.graph = HAN(meta_paths=[['succeed'], ['contain', 'contained']],
                         in_size=config.d_model,
                         hidden_size=config.d_model,
                         out_size=config.d_model,
                         num_heads=[config.num_attention_heads],
                         dropout=config.dropout)
        self.prior_entity_attention_layer = BartAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # self.posterior_entity_attention_layer = BartAttention(
        #     embed_dim=config.d_model,
        #     num_heads=config.encoder_attention_heads,
        #     dropout=config.attention_dropout,
        # )
        self.posterior_entity_attention_layer = self.prior_entity_attention_layer
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.bow_loss = nn.NLLLoss(reduction='mean')
        self.merge_posterior = nn.Linear(3*config.d_model, config.d_model)
        self.merge_prior = nn.Linear(2*config.d_model, config.d_model)
        self.bow_mlp = nn.Linear(config.d_model, config.vocab_size)
        self.config = config
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def build_graph(self, source, target, after_state_id, before_state_id, all_state_repre, sent_node_repre):
        """
        sentences: [sentence_num, 768]
        """
        g = dgl.heterograph({
            ('sentence', 'succeed', 'sentence'): ([i for i in range(sent_node_repre.size()[0] - 1)], [i for i in range(1, sent_node_repre.size()[0])]),
            ('sentence', 'contain', 'entity'): (source, target),
            ('entity', 'contained', 'sentence'): (target, source),
        })
        g = g.to(sent_node_repre.device)
        g.edges['contain'].data['sta'] = torch.stack([all_state_repre[i] for i in (after_state_id if len(after_state_id) > 0 else [0])])
        g.edges['contained'].data['stb'] = torch.stack([all_state_repre[i] for i in (before_state_id if len(before_state_id) > 0 else [0])])
        return g, sent_node_repre

    def graph_encoder(self, source=None, target=None, after_state_id=None, before_state_id=None, all_state=None, encoder_outputs=None):
        all_state_repre = self.get_encoder()(
            input_ids=all_state[0]['input_ids'],
            attention_mask=all_state[0]['attention_mask'],
            return_dict=True,
        )
        all_state_repre = all_state_repre.last_hidden_state.mean(dim=1)
        sent_node_repre = encoder_outputs.last_hidden_state.mean(dim=1)
        graph, _ = self.build_graph(source[0], target[0], after_state_id[0], before_state_id[0], all_state_repre,
                                    sent_node_repre)
        graph_hidden = self.graph(graph, sent_node_repre)
        return graph_hidden

    def attention_layer(self, hidden_states, key_value_states, output_attentions):
        """
        hidden_states [1, tgt_len, d_model]
        key_value_states [1, src_len, d_model]
        output_attentions bool
        """
        logits = torch.bmm(hidden_states, key_value_states.transpose(-1, -2))  # [1, tgt_len, src_len]
        expand_values = key_value_states.unsqueeze(dim=1)  # [1, 1, src_len, d_model]
        # expand_weights = F.gumbel_softmax(logits, tau=0.8, dim=-1, hard=not self.training).unsqueeze(dim=-1)  # [1, tgt_len, src_len, 1]
        expand_weights = F.softmax(logits, dim=-1).unsqueeze(dim=-1)  # [1, tgt_len, src_len, 1]
        states = (expand_values * expand_weights).mean(dim=2)  # [1, tgt_len, d_model]
        return states, logits.squeeze(dim=0), None

    def entity_attention(self, transition_entity, graph_hidden, encoder_hidden, decoder_input_ids=None):
        all_entity_repre = self.get_encoder()(
            input_ids=transition_entity[0]['input_ids'],  # [entity_num, 3]  3是bpe token max_len
            attention_mask=transition_entity[0]['attention_mask'],
            return_dict=True,
        )
        all_entity_repre = all_entity_repre.last_hidden_state.mean(dim=1).unsqueeze(dim=0)  # [1, entity_num, 768]
        encoder_hidden = encoder_hidden.last_hidden_state.mean(dim=1).unsqueeze(dim=0)  # [1, sentence_num, 768]
        graph_hidden = graph_hidden.unsqueeze(dim=0)  # [1, sentence_num, 768*2]

        prior = self.merge_prior(torch.cat([graph_hidden, encoder_hidden], dim=2))
        prior_hidden_states, prior_attn_weights, _ = self.prior_entity_attention_layer(
            hidden_states=prior,
            key_value_states=all_entity_repre,
            output_attentions=True
        )
        # prior_hidden_states, prior_attn_weights, _ = self.attention_layer(hidden_states=prior,
        #     key_value_states=all_entity_repre,
        #     output_attentions=True)
        prior_hidden_states = prior_hidden_states.squeeze(dim=0)  # [sentence_num, 768]
        prior_attn_weights = prior_attn_weights.unsqueeze(dim=0)  # [layer_num, sentence_num, entity_num]
        if decoder_input_ids is not None:
            ground_truth_repre = self.get_encoder()(
                input_ids=decoder_input_ids,
                return_dict=True,
            )
            ground_truth_repre = ground_truth_repre.last_hidden_state.mean(dim=1).unsqueeze(dim=0)  # [1, sentence_num, 768]
            posterior = self.merge_posterior(torch.cat([ground_truth_repre, graph_hidden, encoder_hidden], dim=-1))
            posterior_hidden_states, posterior_attn_weights, _ = self.posterior_entity_attention_layer(
                hidden_states=posterior,
                key_value_states=all_entity_repre,
                output_attentions=True
            )
            # posterior_hidden_states, posterior_attn_weights, _ = self.attention_layer(
            #     hidden_states=posterior,
            #     key_value_states=all_entity_repre,
            #     output_attentions=True
            # )
            posterior_hidden_states = posterior_hidden_states.squeeze(dim=0)  # [sentence_num, 768]
            posterior_attn_weights = posterior_attn_weights.unsqueeze(dim=0)  # [layer_num, sentence_num, entity_num]
            # kl_loss = self.kl_loss(F.log_softmax(prior_attn_weights, dim=-1), F.softmax(posterior_attn_weights, dim=-1).detach())
            kl_loss = self.kl_loss(torch.log(prior_attn_weights), posterior_attn_weights.detach())

            bow_logits = F.log_softmax(self.bow_mlp(posterior_hidden_states), dim=-1)
            seq_len = decoder_input_ids.size(1) - 1
            k_logits = bow_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, self.config.vocab_size)
            bow_loss = self.bow_loss(k_logits, decoder_input_ids[:, 1:].contiguous().view(-1))
            kl_loss += (bow_loss / 10)
            return prior_hidden_states, posterior_hidden_states, kl_loss
        else:
            return prior_hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        source=None,
        target=None,
        after_state_id=None,
        before_state_id=None,
        all_state=None,
        transition_entity=None,
        graph_hidden=None,
        transition_entity_hidden=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if encoder_outputs is None:
            encoder_outputs = self.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if graph_hidden is None:
            graph_hidden = self.graph_encoder(source, target, after_state_id, before_state_id, all_state, encoder_outputs)
        if transition_entity_hidden is None:
            _, transition_entity_hidden, kl_loss = self.entity_attention(transition_entity, graph_hidden, encoder_outputs, decoder_input_ids)
            # transition_entity_hidden = _
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_hidden=graph_hidden,
            transition_entity_hidden=transition_entity_hidden,
            # addi_source_attention_mask=addi_source_attention_mask,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if self.config.use_kl_loss and self.training:
            masked_lm_loss += (kl_loss / 100)

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) -> Dict[str, Any]:
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and (not argument in ['source', 'target', 'after_state_id', 'before_state_id', 'all_state', 'transition_entity'])
        }
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        source, target, after_state_id, before_state_id, all_state, encoder_outputs = \
            model_kwargs['source'],  model_kwargs['target'],  model_kwargs['after_state_id'],  \
            model_kwargs['before_state_id'], model_kwargs['all_state'], model_kwargs['encoder_outputs']
        graph_hidden = self.graph_encoder(source, target, after_state_id, before_state_id, all_state, encoder_outputs)
        model_kwargs["graph_hidden"] = graph_hidden
        prior_hidden_states = self.entity_attention(model_kwargs['transition_entity'], graph_hidden, model_kwargs["encoder_outputs"])
        model_kwargs["transition_entity_hidden"] = prior_hidden_states
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            graph_hidden = None,
            transition_entity_hidden = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

            graph_hidden = graph_hidden.index_select(0, expanded_return_idx.to(graph_hidden.device))
            model_kwargs['graph_hidden'] = graph_hidden

            transition_entity_hidden = transition_entity_hidden.index_select(0, expanded_return_idx.to(transition_entity_hidden.device))
            model_kwargs['transition_entity_hidden'] = transition_entity_hidden
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "graph_hidden": kwargs['graph_hidden'],
            "transition_entity_hidden": kwargs['transition_entity_hidden'],
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class MyCometCallback(TrainerCallback):
    def __init__(self, proj_name, exp_name):
        self._initialized = False
        self.exp = None  # type: MyExperiment
        self.proj_name = proj_name
        self.exp_name = exp_name
        self.exp = MyExperiment(self.proj_name, self.exp_name)
        self.trainer = None  # type: Trainer

    def set_trainer(self, trainer):
        self.trainer = trainer

    def setup(self, args, state, model):
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            self.exp.experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework="transformers")
            pcb_list = [cb for cb in self.trainer.callback_handler.callbacks if isinstance(cb, ProgressCallback)]
            if len(pcb_list) > 0:
                bar = pcb_list[0].training_bar  # type: tqdm.format_dict
                if bar is not None:
                    self.exp.experiment.log_other(key='eta', value=bar.format_interval(
                        (bar.format_dict['total'] - bar.format_dict['n']) / bar.format_dict['rate']))


class AutoDecodeCallback(TrainerCallback):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def auto_decode(self, ckpt_path):
        def run_command(working_path, command, stdout=None):
            working_path = os.path.abspath(working_path)
            if stdout is None:
                with open(os.devnull, 'w') as devnull:
                    child = subprocess.Popen(command, cwd=working_path, shell=True, stdout=devnull)
                    return child
            else:
                child = subprocess.Popen(command, cwd=working_path, shell=True, stdout=stdout)
                return child

        # 创建decode所需flags
        aggs = sys.argv.copy()
        if '--do_train' in aggs:
            aggs.remove('--do_train')
        if '--do_eval' not in aggs:
            aggs.append('--do_eval')
        if '--do_predict' not in aggs:
            aggs.append('--do_predict')
        if '--model_name_or_path' in aggs:
            model_path_index = aggs.index('--model_name_or_path')
            aggs[model_path_index + 1] = ckpt_path
        else:
            aggs.extend(['--model_name_or_path', ckpt_path])
        if '--predict_with_generate' not in aggs:
            aggs.extend(['--predict_with_generate', 'True'])
        aggs.extend(['--disable_tqdm', 'True'])
        flag_str = ' '.join(aggs)

        decode_cmd = ' '.join([sys.executable, flag_str])

        child = run_command(os.path.dirname(os.path.abspath(__file__)), decode_cmd, sys.stdout)
        logging.info('start auto decode %s\n%s' % (ckpt_path, decode_cmd))

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.auto_decode(os.path.join(self.model_dir, f'checkpoint-{state.global_step}'))


@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    model_args: ModelArguments = None
    data_args: DataTrainingArguments = None

    def __call__(self, features):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        max_target_length = self.data_args.max_target_length

        # sentence_entity = [f['sentence_entity'] for f in features]
        samples_text = [f['text'] for f in features]
        samples_summary = [f['summary'] for f in features]
        samples_entity_list = [f['entity_list'] for f in features]
        samples_sentence_entity = [f['sentence_entity'] for f in features]
        for f in features:
            for k in ['adomain', 'qdomain', 'summary', 'token_type_ids', 'retrieval', 'content', 'selftext', 'subreddit',
                      'answers', 'title', 'sentence_entity', 'entity_list', 'text']:
                if k in f:
                    del f[k]
        inputs = []
        for text in samples_text:
            if self.data_args.concat_context:
                inputs.extend([f'{sent} {self.tokenizer.mask_token} {text[sent_i-1] if sent_i != 0 else ""} {text[sent_i+1] if sent_i != (len(text)-1) else ""}' for sent_i, sent in enumerate(text)])
                # inputs.extend([f'{sent} {self.tokenizer.sep_token} {text[sent_i-1] if sent_i != 0 else " "} {self.tokenizer.sep_token} {text[sent_i+1] if sent_i != (len(text)-1) else " "}' for sent_i, sent in enumerate(text)])
            else:
                inputs.extend([sent.replace(' ', '') if self.data_args.chinese_data else sent for sent in text])
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                      truncation=True, add_special_tokens=False)

        targets = []
        for summ in samples_summary:
            targets.extend([sent.replace(' ', '') + ' ' + self.tokenizer.eos_token if self.data_args.chinese_data else sent + ' ' + self.tokenizer.eos_token for sent in summ])
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,
                                    add_special_tokens=False)
        if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        new_features = []
        for inp_ids, inp_attn, lab in zip(model_inputs['input_ids'], model_inputs['attention_mask'], labels["input_ids"]):
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
        to_return = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        def tokenize_entity(e):
            return self.tokenizer(e, max_length=3, padding='max_length', truncation=True, add_special_tokens=False)

        entity_list = [tokenize_entity(elist) for elist in samples_entity_list]
        source = []  # sentences id
        target = []  # entity id
        after_state_id = []
        before_state_id = []

        all_state = []
        all_entity = []
        for batch_idx, sentences_entity in enumerate(samples_sentence_entity):  # for each article in batch (max iter 1)
            for sent in sentences_entity:  # for each sentence
                for ent_obj in sent:  # for entities in sentence (max iter 3)
                    all_state.append(ent_obj['after'])
                    all_state.append(ent_obj['before'])
                    all_entity.append(ent_obj['entity'])
        all_state = [list(set(all_state))]
        all_entity = [list(set(all_entity))]

        all_state_tokenized = [tokenize_entity(slist if len(slist) > 0 else [self.tokenizer.pad_token]) for slist in all_state]
        for batch_idx, sentences_entity in enumerate(samples_sentence_entity):  # for each article in batch (max iter 1)
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
        #
        # batch_ent_list = []
        # batch_sentence_entity_count = []
        # for sentences_entity in samples_sentence_entity:  # for each article in batch (max iter 1)
        #     art_sent_list = []
        #     accu_sentence_entity_count = []
        #     sent_ent_list = []
        #     sent_state_after_list = []
        #     sent_state_before_list = []
        #     for sent in sentences_entity:  # for each sentence
        #         accu_sentence_entity_count.append((len(sent) + accu_sentence_entity_count[-1]) if len(accu_sentence_entity_count) > 0 else len(sent))
        #         for ent_obj in sent:  # for entities in sentence (max iter 3)
        #             sent_ent_list.append(ent_obj['entity'])
        #             sent_state_after_list.append(ent_obj['after'])
        #             sent_state_before_list.append(ent_obj['before'])
        #     batch_sentence_entity_count.append(accu_sentence_entity_count)
        #     sent_ent_list = tokenize_entity(sent_ent_list)
        #     sent_state_after_list = tokenize_entity(sent_state_after_list)
        #     sent_state_before_list = tokenize_entity(sent_state_before_list)
        #     art_sent_list.append({'entity': sent_ent_list, 'after': sent_state_after_list, 'before': sent_state_before_list})
        #     batch_ent_list.append(art_sent_list)
        to_return['transition_entity'] = entity_list
        # to_return['sentence_entity'] = batch_ent_list
        # to_return['accu_sentence_entity_count'] = batch_sentence_entity_count
        return to_return


if __name__ == '__main__':

    path = '/home/gaoshen.gao/pretrain/antbart-ckpt-40000'
    tokenizer = BertTokenizerFast.from_pretrained(path)
    # model = BartForConditionalGeneration.from_pretrained(path)
    model = MyBart.from_pretrained(path)


    TXT = f"周三市场呈现开盘指数小幅高开，盘中银行、券商、房地产等权重板块带动拉升"+tokenizer.eos_token


    input_ids = tokenizer([TXT], return_tensors='pt', add_special_tokens=False)['input_ids']
    print('-------call--------')
    logits = model(input_ids).logits  # type: torch.Tensor
    print(logits.shape)
    print('Greedy --> ', tokenizer.decode(logits[0].softmax(dim=1).argmax(dim=1)))
    print('-------generate--------')

    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    print(tokenizer.decode(summary_ids[0], clean_up_tokenization_spaces=False, skip_special_tokens=True))



