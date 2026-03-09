# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from dataclasses import dataclass
import warnings
from typing import List, Optional, Tuple, Union
import math

import torch.utils.checkpoint
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM, Qwen3ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from peft import LoraConfig, get_peft_model

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from utils_internvl.dataset import get_cus_attn_mask

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


@dataclass
class InternVLChatModelOutput(CausalLMOutputWithPast):
    vit_embeds: Optional[torch.FloatTensor] = None
    ori_vit_embeds: Optional[torch.FloatTensor] = None
    seg_probs: Optional[torch.FloatTensor] = None
    adapt_states: Optional[torch.FloatTensor] = None
    unnorm_last_hidden_state: Optional[torch.FloatTensor] = None

class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer', 
        "Qwen3DecoderLayer",]

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        # config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        config.llm_config._attn_implementation = 'eager'
        # config.llm_config._attn_implementation = 'sdpa'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05, lora_from_layer=0):
        # Determine the target modules based on the architecture of the language model
        if self.config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.config.llm_config.architectures[0] in ['Qwen2ForCausalLM', 'LlamaForCausalLM', 'Qwen3ForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented

        layers_to_transform = None
        if lora_from_layer > 0:
            num_layers = self.language_model.config.num_hidden_layers
            layers_to_transform = list(range(lora_from_layer+1, num_layers))

        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            # modules_to_save=['combine_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM',
            layers_to_transform=layers_to_transform,
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            offset: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds, ori_vit_embeds, cls_embeds = self.extract_feature(pixel_values)
        # vit_embeds, ori_vit_embeds = self.extract_feature_v2(pixel_values)
        ori_vit_embeds = ori_vit_embeds[image_flags == 1].reshape(-1, ori_vit_embeds.shape[-2], ori_vit_embeds.shape[-1])
        vit_embeds = vit_embeds[image_flags == 1].reshape(-1, vit_embeds.shape[-2], vit_embeds.shape[-1])
        vit_batch_size = pixel_values.shape[0]

        # Insert pooling vit embeds to seg pos
        seg_pos = (input_ids == self.seg_token_idx).view(input_embeds.shape[0], -1)
        img_pos = (input_ids == self.img_token_idx).view(input_embeds.shape[0], -1)
        big_patch_indices = torch.cat([torch.zeros(1, device=img_pos.device), (img_pos.sum(dim=1) // self.num_image_token).cumsum(dim=0)]).long()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values_hq=ori_vit_embeds,
            img_pos=img_pos,
            seg_pos=seg_pos,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and self.training:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return InternVLChatModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vit_embeds=vit_embeds,
            ori_vit_embeds=ori_vit_embeds,
            unnorm_last_hidden_state=outputs.unnorm_last_hidden_state if hasattr(outputs, 'unnorm_last_hidden_state') else None,
            seg_probs=outputs.seg_probs if hasattr(outputs, 'seg_probs') else None,
            adapt_states=outputs.adapt_states if hasattr(outputs, 'adapt_states') else None,
        )
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        cls_embeds = vit_embeds[:, :1, :]
        vit_embeds = vit_embeds[:, 1:, :]
        ori_vit_embeds = vit_embeds.clone()

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # (B, H*2, W*2, C)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # vit_embeds = self.mlp1(vit_embeds)

        # ori_vit_embeds = F.interpolate(vit_embeds.view(vit_embeds.shape[0], h//2, w//2, -1).permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        # self shuffle for cls embeds
        # cls_embeds = cls_embeds.expand(-1, int(1/self.downsample_ratio**2), -1).view(cls_embeds.shape[0], int(1/self.downsample_ratio), int(1/self.downsample_ratio), -1).contiguous()
        # cls_embeds = self.pixel_shuffle(cls_embeds, scale_factor=self.downsample_ratio)
        # cls_embeds = cls_embeds.reshape(cls_embeds.shape[0], -1, cls_embeds.shape[-1])

        # ori_vit_embeds = self.conv2x2_shuffle(ori_vit_embeds)
        ori_vit_embeds = self.replicate_shuffle(ori_vit_embeds)
        # ori_vit_embeds = self.interpolate_shuffle(ori_vit_embeds)
        combined_embeds = torch.cat([vit_embeds, ori_vit_embeds], dim=1)
        combined_embeds = self.mlp1(combined_embeds)
        vit_embeds, ori_vit_embeds = combined_embeds.split([vit_embeds.shape[1], ori_vit_embeds.shape[1]], dim=1)

        # ori_vit_embeds = self.mlp1(ori_vit_embeds)
        # ori_vit_embeds = self.mlp2(ori_vit_embeds)
        return vit_embeds, ori_vit_embeds, cls_embeds
    
    def extract_feature_v2(self, pixel_values):
        # h and w are 2x compared to the original image
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        ori_vit_embeds = vit_embeds.clone()

        h = w = int(vit_embeds.shape[1] ** 0.5)
        lh, lw = int(self.num_image_token ** 0.5), int(self.num_image_token ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        # vit_embeds = F.avg_pool2d(vit_embeds.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
        vit_embeds = F.interpolate(vit_embeds.permute(0, 3, 1, 2), size=(lh*2, lw*2), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        # ori_vit_embeds = self.conv2x2_shuffle(ori_vit_embeds)
        ori_vit_embeds = self.replicate_shuffle(ori_vit_embeds)
        ori_vit_embeds = self.mlp1(ori_vit_embeds)
        # ori_vit_embeds = self.mlp2(ori_vit_embeds)
        return vit_embeds, ori_vit_embeds

    def conv2x2_shuffle(self, x):
        """
        将输入特征图通过2x2卷积，stride=1的方式将相邻4个特征拼接
        Input: x shape (B, N, C)
        Output: shape (B, N, C*4)
        """
        b, n, c = x.shape
        h, w = int(n ** 0.5), int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        
        # x_padded = F.pad(x, (0, 1, 0, 1), mode='replicate')  # (B, C, H+1, W+1)
        x_unfolded = x.unfold(2, 2, 1)  # (B, C, H, W+1, 2)
        x_unfolded = x_unfolded.unfold(3, 2, 1)  # (B, C, H, W, 2, 2)
        x_unfolded = x_unfolded.contiguous().view(b, c, h-1, w-1, 4)  # (B, C, H, W, 4)
        x_concat = x_unfolded.permute(0, 4, 1, 2, 3).contiguous()  # (B, 4, C, H, W)
        # x_concat = x_concat.reshape(b, c * 4, h, w)  # (B, C*4, H, W)
        x_concat = x_concat.reshape(b, c * 4, h-1, w-1)  # (B, C*4, H, W)
        x_concat = F.interpolate(x_concat, size=(h, w), mode="bilinear", align_corners=False)
        
        return x_concat.flatten(2).transpose(1, 2) # (B, N, C*4)

    def replicate_shuffle(self, x):
        """
        Replicate the input feature map by 2x2
        Input: x shape (B, N, C)
        Output: shape (B, N, C*4)
        """
        b, n, c = x.shape
        h, w = int(n ** 0.5), int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = x.repeat(1, 4, 1, 1)
        return x.flatten(2).transpose(1, 2) # (B, N, C*4)

    def interpolate_shuffle(self, x):
        """
        Interpolate the input feature map by 2x2
        Input: x shape (B, N, C)
        Output: shape (B, N, C*4)
        """
        b, n, c = x.shape
        h, w = int(n ** 0.5), int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = F.interpolate(x, size=(h*2, w*2), mode="bilinear", align_corners=False)
        x = F.pixel_unshuffle(x, downscale_factor=2) # (B, C*4, H, W)
        return x.flatten(2).transpose(1, 2) # (B, N, C*4)

    def apply_2d_sincos_pos_embed(self, vit_embeds, h=None, w=None):
        """
        Apply 2D sin-cos positional encoding to image tokens efficiently.
        
        Args:
            vit_embeds: image embeddings [B, N, C] where N = h*w
            h, w: optional height and width, auto-calculated if None
            
        Returns:
            vit_embeds_with_pos: image embeddings with positional encoding [B, N, C]
        """
        B, N, C = vit_embeds.shape
        assert C % 2 == 0, "Embedding dimension must be even for sin-cos encoding"
        
        # Auto-calculate h, w if not provided
        if h is None or w is None:
            h = w = int(N ** 0.5)
            assert h * w == N, f"Cannot auto-calculate square grid for N={N}"
        
        device, dtype = vit_embeds.device, vit_embeds.dtype
        
        # Generate position coordinates efficiently
        y_pos = torch.arange(h, dtype=dtype, device=device).unsqueeze(1).expand(h, w).flatten()  # [N]
        x_pos = torch.arange(w, dtype=dtype, device=device).unsqueeze(0).expand(h, w).flatten()  # [N]
        
        # Generate frequency bands
        dim_half = C // 4  # Each coordinate uses half dimensions
        omega = torch.arange(dim_half, dtype=dtype, device=device)
        omega = 1.0 / (10000 ** (omega / dim_half))  # [dim_half]
        
        # Compute sin-cos embeddings for both dimensions
        y_embed = torch.einsum('n,d->nd', y_pos, omega)  # [N, dim_half]
        x_embed = torch.einsum('n,d->nd', x_pos, omega)  # [N, dim_half]
        
        # Combine sin and cos for both x and y
        pos_embed = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),  # y coordinates
            torch.sin(x_embed), torch.cos(x_embed)   # x coordinates
        ], dim=1)  # [N, C]
        
        # Add to embeddings
        return vit_embeds + pos_embed.unsqueeze(0)  # Broadcasting: [B, N, C] + [1, N, C]

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        # img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # selected = (model_inputs['input_ids'] == img_context_token_id)
        # if selected.any():
        #     B, N = input_ids.shape
        #     fake_embed = torch.zeros(B, N, 1)
        #     text_mask = ~model_inputs['attention_mask']
        #     custom_causal_mask = get_cus_attn_mask(fake_embed, text_mask, selected)
        #     print(custom_causal_mask.shape)

        attention_mask = model_inputs['attention_mask'].to(self.device)
        # attention_mask = custom_causal_mask.to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, ori_vit_embeds, _ = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings