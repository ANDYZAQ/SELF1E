import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import numpy_pad_and_concatenate

from model.internvl3.modeling_internvl_chat import InternVLChatModel
from utils_global.losses import dice_loss, sigmoid_ce_loss
from typing import Optional, Tuple, Union, List
import math
from model.injector.Qwen3Res import QwenCausalInject as Qwen3CausalInject
from model.injector.QwenRes import QwenCausalInject as Qwen2CausalInject  

class InternVL3SELF1E(InternVLChatModel):
    def __init__(self, config, **kwargs):
        # 提取需要的参数
        out_dim = kwargs.pop("out_dim", None)
        ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        use_mm_start_end = kwargs.pop("use_mm_start_end", None)
        img_token_idx = kwargs.pop("img_token_idx", None)

        # 可以在这里使用这些参数，例如保存为类的属性
        self.out_dim = out_dim
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.seg_token_idx = seg_token_idx
        self.use_mm_start_end = use_mm_start_end
        self.img_token_idx = img_token_idx
        # self.seg_token = kwargs.pop("seg_token", None)

        super().__init__(config, **kwargs)

        self.combine_decode_proj = nn.Sequential(
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size)
        )
        self.combine_decode_proj_all = nn.Sequential(
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size * int(1 / self.downsample_ratio) ** 2)
        )
        self.combine_decode_proj_ori = nn.Sequential(
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size * int(1 / self.downsample_ratio) ** 2)
        )
        self.language_model = Qwen3CausalInject(config.llm_config) if config.llm_config.architectures[0] == 'Qwen3ForCausalLM' else Qwen2CausalInject(config.llm_config)
        
        self._tp_plan = {}

    def pixel_unshuffle(self, x, scale_factor=2.0):
        """
        Reverse operation of pixel_shuffle. 
        Converts from larger spatial dimensions to smaller spatial dimensions with more channels.
        This is the inverse of pixel_shuffle with scale_factor=0.5.
        
        Args:
            x: Input tensor of shape (N, W, H, C)
            scale_factor: Factor to reduce spatial dimensions by (default 2.0, inverse of pixel_shuffle's 0.5)
        
        Returns:
            Output tensor with reduced spatial dimensions and increased channels
        """
        n, w, h, c = x.size()
        
        # Reverse the final permute of pixel_shuffle (if not v1)
        if hasattr(self, 'ps_version') and self.ps_version == 'v1':
            # For v1, height and width were not swapped back in pixel_shuffle
            pass
        else:
            # N, W, H, C --> N, H, W, C (undo the final permute of pixel_shuffle)
            x = x.permute(0, 2, 1, 3).contiguous()
            w, h = h, w  # Update dimensions after permute
        
        # Reverse step 3: N, H, W, C --> N, H, W // scale, C * scale
        x = x.view(n, h, int(w / scale_factor), int(c * scale_factor))
        # Reverse step 2: N, H, W // scale, C * scale --> N, W // scale, H, C * scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # Reverse step 1: N, W // scale, H, C * scale --> N, W // scale, H // scale, C * (scale^2)
        x = x.view(n, int(w / scale_factor), int(h / scale_factor), int(c * (scale_factor * scale_factor)))
        
        return x

    def forward(self, 
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        masks_list: Optional[torch.LongTensor] = None,
        target_aspect_ratios_list: Optional[List] = None,
        inference: bool = False,
        offset: Optional[torch.LongTensor] = None,
        **kwargs):
        return_dict = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            labels=labels,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        selected = (input_ids == self.img_context_token_id)
        # last_hidden_state = return_dict.hidden_states[-1]
        # last_hidden_state = return_dict.hidden_states[-1]
        last_hidden_state = return_dict.unnorm_last_hidden_state
        # adapt_states = torch.stack(return_dict.adapt_states, dim=0).sum(dim=0) # b, N, c
        # adapt_image_embeds = adapt_states[selected]

        llm_last_image_embeds = last_hidden_state[selected]
        image_embeds = return_dict.ori_vit_embeds
        vit_embeds = return_dict.vit_embeds
        # image_embeds = self.img_proj(image_embeds)

        assert torch.isnan(image_embeds).sum() == 0, "image_embeds contains nan"

        # summarize the dividing points of the image
        image_pivots = selected.sum(dim=1).cumsum(dim=0)
        image_pivots = torch.cat([torch.zeros_like(image_pivots[:1]), image_pivots], dim=0)
        num_big_patches_per_sample = (selected.sum(dim=1) // self.num_image_token).to(torch.long)
        big_patch_pivots = torch.cat([
            torch.zeros(1, dtype=torch.long, device=selected.device),
            num_big_patches_per_sample.cumsum(dim=0)
        ])

        ce_loss = return_dict.loss

        mask_bce_loss = torch.zeros(1, device=image_embeds.device)
        mask_dice_loss = torch.zeros(1, device=image_embeds.device)
        seg_loss = torch.zeros(1, device=image_embeds.device)
        mse_loss = torch.zeros(1, device=image_embeds.device)
        pred_masks = []
        for bs in range(len(input_ids)):
            # targeting image embeds and seg embeds
            n_big_patches = num_big_patches_per_sample[bs]
            llm_image_embeds = llm_last_image_embeds[image_pivots[bs]:image_pivots[bs + 1]] #.view(n_big_patches, self.num_image_token, -1) # n_big, N, C
            curr_image_embeds = image_embeds[big_patch_pivots[bs]:big_patch_pivots[bs + 1]] # n_big, N, C
            seg_pos = torch.cat([(input_ids[bs] == self.seg_token_idx)[1:], torch.zeros_like((input_ids[bs] == self.seg_token_idx)[:1]).bool()], dim=0)
            if seg_pos.sum() == 0:
                pred_masks.append(torch.zeros(0, 1, 1, device=input_ids.device))
                continue
            seg_embed = last_hidden_state[bs][seg_pos] # nseg * C

            # RFA + RFR
            llm_image_embeds = self.combine_decode_proj_all(llm_image_embeds)
            llm_image_embeds = self.reshape_probs(llm_image_embeds, target_aspect_ratios_list[bs]) # C, H, W
            vit_image_embeds = vit_embeds[big_patch_pivots[bs]:big_patch_pivots[bs + 1]] # n_big, N, C
            vit_image_embeds = self.combine_decode_proj_ori(vit_image_embeds)
            vit_image_embeds = self.reshape_probs(vit_image_embeds.flatten(0, 1), target_aspect_ratios_list[bs]) # C, H, W
            assert llm_image_embeds.shape == vit_image_embeds.shape, f"llm_image_embeds.shape: {llm_image_embeds.shape}, vit_image_embeds.shape: {vit_image_embeds.shape}"
            curr_adapt_image_embeds = llm_image_embeds - vit_image_embeds # C, H, W
            curr_adapt_image_embeds = self.pixel_unshuffle(curr_adapt_image_embeds.permute(1, 2, 0).unsqueeze(0), self.downsample_ratio).squeeze(0) # H, W, C
            
            curr_image_embeds = self.reshape_probs(curr_image_embeds.flatten(0, 1), target_aspect_ratios_list[bs]) # c, h, w
            curr_image_embeds = curr_image_embeds.permute(1, 2, 0)
            curr_image_embeds = self.combine_decode_proj_ori(curr_image_embeds)  # h, w, C
            curr_image_embeds = self.pixel_unshuffle(curr_image_embeds.unsqueeze(0), self.downsample_ratio).squeeze(0) # h, w, C

            curr_adapt_image_embeds = F.interpolate(curr_adapt_image_embeds.permute(2, 0, 1).unsqueeze(0), size=curr_image_embeds.shape[:2], mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0) # c, h, w
            # curr_image_embeds = F.avg_pool2d(curr_image_embeds.permute(2, 0, 1).unsqueeze(0), kernel_size=2, stride=2).squeeze(0).permute(1, 2, 0) # h, w, c
            # print(curr_image_embeds.shape, curr_adapt_image_embeds.shape)
            curr_image_embeds = curr_image_embeds + curr_adapt_image_embeds#.permute(1, 2, 0) # h, w, c
            curr_image_embeds = self.combine_decode_proj(curr_image_embeds)

            seg_embed = self.combine_decode_proj_all(seg_embed) # nseg, C
            seg_embed = self.pixel_unshuffle(seg_embed.unsqueeze(1).unsqueeze(1), self.downsample_ratio).mean(dim=(1, 2)) # nseg, C
            seg_embed = self.combine_decode_proj(seg_embed) # nseg, C

            # Final prediction with attention-like operation
            seg_prob = torch.matmul(curr_image_embeds, seg_embed.transpose(-1, -2)).permute(2, 0, 1) # nseg, h, w
            seg_prob = seg_prob / math.sqrt(curr_image_embeds.shape[-1])

            pred_masks.append(seg_prob)
            
            if not inference and masks_list is not None and seg_prob.shape[0]:
                llm_image_embeds = self.pixel_unshuffle(llm_image_embeds.permute(1, 2, 0).unsqueeze(0), self.downsample_ratio).squeeze(0) # h, w, C
                seg_prob_low = torch.matmul(llm_image_embeds, seg_embed.transpose(-1, -2)).permute(2, 0, 1) # nseg, h, w
                seg_prob_low = seg_prob_low / math.sqrt(llm_image_embeds.shape[-1])
                
                seg_gt = masks_list[bs].to(seg_prob.device)
                h, w = seg_gt.shape[-2:]
                if min(h, w) > 128:
                    # resize for training efficiency
                    h, w = h * 128 // min(h, w), w * 128 // min(h, w)
                    seg_gt = F.interpolate(seg_gt.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1) # nseg, h, w
                # reshape seg_prob to ratio of h and w
                seg_prob = F.interpolate(seg_prob.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1) # nseg, h, w

                mask_bce_loss += sigmoid_ce_loss(seg_prob, seg_gt, num_masks=seg_gt.shape[0]) * seg_gt.shape[0]
                mask_dice_loss += dice_loss(seg_prob, seg_gt, num_masks=seg_gt.shape[0]) * seg_gt.shape[0]
                seg_prob_low = F.interpolate(seg_prob_low.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1) # nseg, h, w
                mask_bce_loss += sigmoid_ce_loss(seg_prob_low, seg_gt, num_masks=seg_gt.shape[0]) * seg_gt.shape[0]
                mask_dice_loss += dice_loss(seg_prob_low, seg_gt, num_masks=seg_gt.shape[0]) * seg_gt.shape[0]


        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }

        mask_bce_loss = mask_bce_loss / len(input_ids)
        mask_dice_loss = mask_dice_loss / len(input_ids)
        seg_loss = self.bce_loss_weight * mask_bce_loss + self.dice_loss_weight * mask_dice_loss
        loss = self.ce_loss_weight * ce_loss + seg_loss        

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": seg_loss,
            "other_loss": mse_loss,
        }

    def reshape_probs(self, seg_prob, target_aspect_ratios, proj_ratio=None):
        """
        seg_prob: N, nseg
        """
        if proj_ratio is None:
            proj_ratio = seg_prob.shape[0] / self.num_image_token / target_aspect_ratios[0] / target_aspect_ratios[1]
        assert seg_prob.shape[0] % int(self.num_image_token*proj_ratio) == 0, f"seg_prob.shape[0] % (self.num_image_token * proj_ratio) != 0, {seg_prob.shape[0]} % {self.num_image_token * proj_ratio} != 0"
        num_patches = seg_prob.shape[0] // int(self.num_image_token * proj_ratio)
        h, w = int((self.num_image_token * proj_ratio) ** 0.5), int((self.num_image_token * proj_ratio) ** 0.5)
        # dynamic_preprocess returns target_aspect_ratio as (cols, rows).
        # We must use (rows, cols) when reshaping to [rows*h, cols*w].
        patch_rows, patch_cols = target_aspect_ratios[1], target_aspect_ratios[0]
        # print(f"seg_prob.shape: {seg_prob.shape}, num_patches: {num_patches}, h: {h}, w: {w}, rows: {patch_rows}, cols: {patch_cols}")
        seg_prob = seg_prob.view(num_patches, int(self.num_image_token * proj_ratio), -1) # nimg, N // nimg, nseg
        seg_prob = seg_prob.view(patch_rows, patch_cols, h, w, -1) # rows, cols, h, w, nseg
        seg_prob = seg_prob.permute(0, 2, 1, 3, 4)
        seg_prob = seg_prob.flatten(0,1).flatten(1,2).permute(2, 0, 1) # nseg, h, w
        return seg_prob