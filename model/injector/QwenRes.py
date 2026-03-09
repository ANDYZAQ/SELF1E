import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer, Cache, FlashAttentionKwargs, Unpack, BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import DynamicCache
from dataclasses import dataclass
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask, TransformersKwargs

logger = logging.get_logger(__name__)

class Qwen2DecoderLayerInject(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask.to(dtype=hidden_states.dtype) if attention_mask is not None else None,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        adapt_states = hidden_states.clone()
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        adapt_states = adapt_states + hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (adapt_states, )

        return outputs


class Qwen2ModelInject(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        # self.layers = nn.ModuleList([Qwen2DecoderLayerInject(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        img_pos: Optional[torch.LongTensor] = None,
        seg_pos: Optional[torch.LongTensor] = None,
        seg_img_embed: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create initial position embeddings - will be updated dynamically per layer
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_adapt_states = ()  # Collect adapt states from all layers

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # for i in range(seg_pos.shape[0]):
            #     if seg_pos[i].sum() > 0:
            #         hidden_states[i, seg_pos[i]] = hidden_states[i, seg_pos[i]] + seg_img_embed.repeat(seg_pos[i].sum(), 1) * (1/self.config.num_hidden_layers)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type].to(dtype=hidden_states.dtype) if causal_mask_mapping[decoder_layer.attention_type] is not None else None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                # output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # hidden_states = layer_outputs[0]
            hidden_states = layer_outputs

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)
            
            # Extract adapt_states (always the last element in layer_outputs)
            # adapt_states = layer_outputs[-1]
            adapt_states = None
            all_adapt_states += (adapt_states,)

        unnorm_last_hidden_state = hidden_states.clone()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # output = BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values if use_cache else None,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        output = CheckOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            adapt_states=all_adapt_states,  # Add adapt states to output
            unnorm_last_hidden_state=unnorm_last_hidden_state,
        )
        return output if return_dict else output.to_tuple()
    

@dataclass
class CheckOutput(BaseModelOutputWithPast):
    adapt_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # Add adapt_states field
    unnorm_last_hidden_state: Optional[torch.FloatTensor] = None

class QwenCausalInject(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)  # Initialize with full config
        self.model = Qwen2ModelInject(config)  # Use the custom model with injection

    def forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values_hq: Optional[torch.FloatTensor] = None,
        img_pos: Optional[torch.LongTensor] = None,
        seg_pos: Optional[torch.LongTensor] = None,
        seg_img_embed: Optional[torch.FloatTensor] = None,
        **kwargs):
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
            cache_position=cache_position,
            pixel_values_hq=pixel_values_hq,
            img_pos=img_pos,
            seg_pos=seg_pos,
            seg_img_embed=seg_img_embed,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return QwenCausalInjectOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            adapt_states=outputs.adapt_states if hasattr(outputs, 'adapt_states') else None, # Add adapt_states to output
            unnorm_last_hidden_state=outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        )
    

@dataclass
class QwenCausalInjectOutput(CausalLMOutputWithPast):
    adapt_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # Add adapt_states field
    unnorm_last_hidden_state: Optional[torch.FloatTensor] = None
