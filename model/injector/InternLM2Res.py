import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from dataclasses import dataclass
from ..internvl2.modeling_internlm2 import (
    InternLM2DecoderLayer, 
    InternLM2Model, 
    InternLM2ForCausalLM,
    InternLM2Config
)

logger = logging.get_logger(__name__)

class InternLM2DecoderLayerInject(InternLM2DecoderLayer):
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        # Store adapt states after attention but before residual connection
        adapt_states = hidden_states.clone()
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        
        # Add MLP output to adapt states
        adapt_states = adapt_states + hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        # Always add adapt_states as the last element
        outputs += (adapt_states,)

        return outputs


class InternLM2ModelInject(InternLM2Model):
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        # Replace layers with inject versions
        self.layers = nn.ModuleList([InternLM2DecoderLayerInject(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.attn_implementation == 'flash_attention_2':
            from ..internvl2.modeling_internlm2 import _import_flash_attn
            _import_flash_attn()

        # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        if inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        if self.config.attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            if attention_mask.dim() != 4:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_adapt_states = ()  # Collect adapt states from all layers
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                if output_attentions:
                    next_decoder_cache += (layer_outputs[2],)
                else:
                    next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Extract adapt_states (always the last element in layer_outputs)
            adapt_states = layer_outputs[-1]
            all_adapt_states += (adapt_states,)

        unnorm_last_hidden_state = hidden_states.clone()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            output = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return output + (all_adapt_states,)
            
        return InternLM2ModelInjectOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            adapt_states=all_adapt_states,  # Add adapt states to output
            unnorm_last_hidden_state=unnorm_last_hidden_state,
        )


@dataclass
class InternLM2ModelInjectOutput(BaseModelOutputWithPast):
    adapt_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # Add adapt_states field
    unnorm_last_hidden_state: Optional[torch.FloatTensor] = None

class InternLM2CausalInject(InternLM2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)  # Initialize with full config
        self.model = InternLM2ModelInject(config)  # Use the custom model with injection

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values_hq: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        return InternLM2CausalInjectOutput(
            loss=loss,
            logits=logits.to(device),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            adapt_states=outputs.adapt_states if hasattr(outputs, 'adapt_states') else None, # Add adapt_states to output
            unnorm_last_hidden_state=outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        )


@dataclass
class InternLM2CausalInjectOutput(CausalLMOutputWithPast):
    adapt_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # Add adapt_states field 
    unnorm_last_hidden_state: Optional[torch.FloatTensor] = None