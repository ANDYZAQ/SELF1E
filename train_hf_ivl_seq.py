# this is a test
import argparse
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments
from transformers.integrations.deepspeed import deepspeed_optim_sched
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
from accelerate.utils import DistributedType
from safetensors.torch import load_file
from transformers.utils.import_utils import is_sagemaker_mp_enabled

from model.internvl3.modeling_internvl_chat import InternVLChatConfig
from model.InternVL3_self1e import InternVL3SELF1E
from utils_internvl.dataset import HybridDatasetSequential, ValDataset, collate_fn
from utils_internvl.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, 
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils_global.utils_global import get_ds_config, find_linear_layers_qwen, _freeze_params, get_all_eval_datasets, evaluate_all_datasets
from utils_global.training_args import get_training_args_seq

from utils_global.visualization import visualize_attention, visualize_entropy, visualize_mask, visualize_img_cont_attention, visualize_img_cont_attention_var
import json
import warnings
warnings.filterwarnings("ignore")


def load_weight_safetensors(weight_dir):
    """Load safetensors weights, supporting both single file and sharded files.
    
    Args:
        weight_dir: Directory containing model weights
        
    Returns:
        state_dict: Loaded state dictionary
    """
    weight_path = f"{weight_dir}/model.safetensors"
    if os.path.exists(weight_path):
        return load_file(weight_path)
    
    shard_files = sorted(glob.glob(f"{weight_dir}/model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"Could not find model.safetensors or model-*.safetensors in {weight_dir}")
    
    state_dict = {}
    for shard_file in shard_files:
        shard_dict = load_file(shard_file)
        state_dict.update(shard_dict)
        print(f"Loaded: {os.path.basename(shard_file)}")
    
    print(f"Loaded {len(shard_files)} weight shards from {weight_dir}.")
    return state_dict


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--model_weights", default=None, type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        # default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        default="ade20k||cocostuff||pascal_part||paco_lvis",
        # default="cocostuff",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--vision_lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)
    parser.add_argument("--lora_from_layer", default=0, type=int, help="Apply LoRA from this layer onwards. Default is 0, meaning all layers.")
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--use_llm_lora", action="store_true", default=False)
    parser.add_argument("--use_vision_lora", action="store_true", default=False)
    parser.add_argument("--optimize_vision", action="store_true", default=False)

    parser.add_argument("--high_res", action="store_true", default=False)
    parser.add_argument("--eval_all_datasets", action="store_true", default=False, help="Evaluate on all datasets after training or as standalone evaluation")
    parser.add_argument("--eval_all_only", action="store_true", default=False, help="Only evaluate on all datasets without training")
    parser.add_argument("--eval_visualize", action="store_true", default=False, help="Visualize the evaluation results")
    return parser.parse_args(args)


class LISATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.writer = kwargs.pop('writer', None)
        self.args_cmd = kwargs.pop('args_cmd', None)
        super().__init__(*args, **kwargs)
        
        # Initialize metrics
        self.batch_time = AverageMeter("Time", ":6.3f")
        self.data_time = AverageMeter("Data", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4f")
        self.ce_losses = AverageMeter("CeLoss", ":.4f")
        self.mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
        self.mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
        self.mask_losses = AverageMeter("MaskLoss", ":.4f")
        self.other_losses = AverageMeter("OtherLoss", ":.4f")
        
        self.progress = ProgressMeter(
            self.args_cmd.steps_per_epoch,
            [
                # self.batch_time,
                self.losses,
                self.ce_losses,
                self.mask_losses,
                self.mask_bce_losses,
                self.mask_dice_losses,
                self.other_losses,
            ],
            prefix="Epoch: [{}]".format(self.state.epoch),
        )

    def create_optimizer_and_scheduler(self, num_training_steps):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        decay_parameters = self.get_decay_parameter_names(opt_model)
        # Remove lora parameters from decay_parameters
        decay_parameters = [n for n in decay_parameters if "lora" not in n]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "language_model" in n)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "language_model" not in n)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args_cmd.vision_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "language_model" in n)
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "language_model" not in n)
                ],
                "weight_decay": 0.0,
                "lr": self.args_cmd.vision_lr,
            },
        ]
        
        self.optimizer, self.lr_scheduler = deepspeed_optim_sched(
            self, self.accelerator.state.deepspeed_plugin.hf_ds_config, self.args, num_training_steps, optimizer_grouped_parameters
        )

        return self.lr_scheduler
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        
        loss = outputs["loss"]
        ce_loss = outputs["ce_loss"]
        mask_bce_loss = outputs["mask_bce_loss"]
        mask_dice_loss = outputs["mask_dice_loss"]
        mask_loss = outputs["mask_loss"]
        other_loss = outputs["other_loss"] if "other_loss" in outputs else torch.zeros(1, device=inputs["input_ids"].device)
        # batch_size = inputs["images"].size(0)
        batch_size = inputs["input_ids"].shape[0]

        self.losses.update(loss.item(), batch_size)
        self.ce_losses.update(ce_loss.item(), batch_size)
        self.mask_bce_losses.update(mask_bce_loss.item(), batch_size)
        self.mask_dice_losses.update(mask_dice_loss.item(), batch_size)
        self.mask_losses.update(mask_loss.item(), batch_size)
        self.other_losses.update(other_loss.item(), batch_size)
        # if dist.is_initialized():
        #     print(f"dist.get_rank(): {dist.get_rank()} loss: {self.losses.sum}, ce_loss: {self.ce_losses.sum}, mask_bce_loss: {self.mask_bce_losses.sum}, mask_dice_loss: {self.mask_dice_losses.sum}, mask_loss: {self.mask_losses.sum}")
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        
        if self.accelerator.is_main_process and self.state.global_step % self.args_cmd.print_freq == 0:
            self.progress.prefix="Epoch: [{}]".format(self.state.epoch)
            self.progress.display(self.state.global_step % self.args_cmd.steps_per_epoch + 1)
            
            if self.writer is not None:
                self.writer.add_scalar("train/loss", self.losses.avg, self.state.global_step)
                self.writer.add_scalar("train/ce_loss", self.ce_losses.avg, self.state.global_step)
                self.writer.add_scalar("train/mask_bce_loss", self.mask_bce_losses.avg, self.state.global_step)
                self.writer.add_scalar("train/mask_dice_loss", self.mask_dice_losses.avg, self.state.global_step)
                self.writer.add_scalar("train/mask_loss", self.mask_losses.avg, self.state.global_step)
                self.writer.add_scalar("metrics/total_secs_per_batch", self.batch_time.avg, self.state.global_step)
                self.writer.add_scalar("metrics/data_secs_per_batch", self.data_time.avg, self.state.global_step)
                
            # Reset metrics
            self.batch_time.reset()
            self.data_time.reset()
            self.losses.reset()
            self.ce_losses.reset()
            self.mask_bce_losses.reset()
            self.mask_dice_losses.reset()
            self.mask_losses.reset()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Your custom validation logic
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        
        self.model.eval()
        torch.cuda.empty_cache()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        idx = 0
        for input_dict in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)
            if self.args_cmd.precision == "fp16":
                input_dict["pixel_values"] = input_dict["pixel_values"].half()
            elif self.args_cmd.precision == "bf16":
                input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
            else:
                input_dict["pixel_values"] = input_dict["pixel_values"].float()
            conversation_list = input_dict["conversation_list"]
            
            with torch.no_grad():
                output_dict = self.model(**input_dict)
            
            pred_masks = output_dict["pred_masks"]
            # masks_list = output_dict["gt_masks"][0].int()
            masks_list = output_dict["gt_masks"]
            soft_masks = pred_masks[0][0].clone()
            pred_masks = [F.interpolate(pred_mask.unsqueeze(0), size=mask_list.shape[-2:], mode="bilinear", align_corners=False).squeeze(0) for pred_mask, mask_list in zip(pred_masks, masks_list)]
            output_list = [(pred_mask > 0).int() for pred_mask in pred_masks]
            assert len(output_list) == len(masks_list) == len(conversation_list)
            
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for i, (mask_i, output_i) in enumerate(zip(masks_list, output_list)):
                idx += 1
                mask_i = mask_i.long()
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
                if self.args_cmd.eval_visualize and self.train_dataset is None and acc_iou[1] < 0.8:
                    # only visualize during eval only
                    visualize_mask(input_dict["image_paths"][0], output_i[0], mask_i[0], conversation_list[i], f"{self.args_cmd.log_dir}/figs/", idx, (intersection_i/union_i)[1].item(), self.args_cmd.val_dataset)
                    
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / len(masks_list)
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=len(masks_list))
        
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]
        
        metrics = {
            f"{metric_key_prefix}_giou": float(giou),
            f"{metric_key_prefix}_ciou": float(ciou),
        }

        if self.accelerator.is_main_process and self.writer is not None:
            self.writer.add_scalar("val/giou", giou, self.state.epoch)
            self.writer.add_scalar("val/ciou", ciou, self.state.epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

            metrics_file = os.path.join(self.args_cmd.log_dir, 'evaluation_metrics.json')
            try:
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_metrics = []

            epoch_metrics = {
                'val_dataset': self.args_cmd.val_dataset,
                'epoch': self.state.epoch,
                **metrics
            }
            existing_metrics.append(epoch_metrics)

            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=4)

        self.log(metrics)
        return metrics


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    
    # Handle eval_all_only mode
    if args.eval_all_only:
        args.eval_only = True
        args.eval_all_datasets = True
    
    # Prepare training arguments
    training_args = get_training_args_seq(args)
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    assert dist.is_initialized()
    print(f"Current device: {dist.get_rank()}")
    if dist.get_rank() in [-1, 0]:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True, 
        trust_remote_code=True,
    )
    # tokenizer = processor.tokenizer
    # tokenizer.pad_token = tokenizer.eos_token
    num_added_tokens = tokenizer.add_tokens(["[SEG]"])
    assert "[SEG]" in tokenizer.get_vocab()
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    # I forget to remove this code before training of most official versions
    # If you want to train by yourself, you can remove the following code
    num_added_tokens = tokenizer.add_tokens(["[SEGT]"])
    assert "[SEGT]" in tokenizer.get_vocab()
    args.segt_token_idx = tokenizer.convert_tokens_to_ids("[SEGT]")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "use_mm_start_end": args.use_mm_start_end,
        "img_token_idx": tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    config = InternVLChatConfig.from_pretrained(args.version)
    model = InternVL3SELF1E.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_flash_attn=True,
        device_map=args.local_rank, 
        # trust_remote_code=True,
        config=config,
        **model_args
    )
    print(f"Finish loading MLLM from {args.version}.")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.img_context_token_id = torch.tensor(tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN))

    model.language_model.resize_token_embeddings(len(tokenizer))

    if not args.eval_only:
        # model.language_model.lm_head.requires_grad = True
        try:
            for name, param in model.named_parameters():
                if "combine_proj" in name:
                    print(f"Initializing {name}")
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                elif "seg_norm" in name or "img_norm" in name:
                    print(f"Initializing {name}")
                    if "weight" in name:
                        nn.init.ones_(param)
                    elif "variance_epsilon" in name:
                        nn.init.zeros_(param)
        except Exception as e:
            print(f"Error initializing {name}: {e}")
    if args.model_weights is not None:
        model.load_state_dict(load_weight_safetensors(args.model_weights), strict=False)
        print(f"Finish loading model weights from {args.model_weights}.")
    
    if not args.eval_only:
        if args.use_llm_lora:
            # _freeze_params(model.language_model)
            model.wrap_llm_lora(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_from_layer=args.lora_from_layer)
        if not args.optimize_vision:
            _freeze_params(model.vision_model)
            _freeze_params(model.mlp1)
        elif args.use_vision_lora:
            _freeze_params(model.vision_model)
            model.wrap_backbone_lora(r=args.lora_r, lora_alpha=2 * args.lora_r)
        else:
            pass
    
    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "combine_", "seg_norm", "output"]
            ]
        ):
            p.requires_grad = True
    if args.lora_from_layer > 0:
        _freeze_params(model.language_model.base_model.model.model.seg_layers[0])
        _freeze_params(model.seg_img_embed)
    if not args.eval_only:
        for n, p in model.named_parameters():
            if p.requires_grad:
                if dist.get_rank() == 0:
                    print(f"Training {n}")

    # Prepare datasets
    world_size = torch.cuda.device_count()
    if args.eval_only:
        train_dataset = None
    else:
        train_dataset = HybridDatasetSequential(
            args.dataset_dir,
            tokenizer,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            use_high_res=args.high_res,
            bsz=args.batch_size,
        )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.val_dataset,
            args.image_size,
            use_high_res=args.high_res,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples." if train_dataset is not None else f"Validating with {len(val_dataset)} examples."
        )
    elif args.eval_all_datasets:
        # For eval_all_datasets mode, we'll create val_dataset dynamically in evaluate_all_datasets
        val_dataset = None
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    if args.eval_only:
        # load peft weights and merge
        # resume_from_checkpoint = get_last_checkpoint(args.log_dir)
        resume_from_checkpoint = os.path.join(args.log_dir, "ckpt_model")
        print(f"Loading checkpoint: {resume_from_checkpoint}")
        model.load_state_dict(load_weight_safetensors(resume_from_checkpoint), strict=False)
        model.eval()
    elif args.auto_resume:
        resume_from_checkpoint = get_last_checkpoint(args.log_dir)
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = args.resume

    # Create trainer
    trainer = LISATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer, 
            use_mm_start_end=args.use_mm_start_end,
            num_image_token=model.num_image_token, 
            local_rank=dist.get_rank(),
        ),
        writer=writer,
        args_cmd=args,
    )

    # Training
    if args.eval_only:
        if args.eval_all_datasets:
            # Evaluate on all datasets
            evaluate_all_datasets(trainer, args, tokenizer)
        else:
            # Single dataset evaluation
            trainer.evaluate()
    else:
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if dist.get_rank() == 0:
            if args.use_llm_lora:
                trainer.model.language_model.merge_and_unload()
                trainer.model.language_model = trainer.model.language_model.model
                trainer.model.config.use_llm_lora = 0
            if args.use_vision_lora:
                trainer.model.vision_model.merge_and_unload()
                trainer.model.vision_model = trainer.model.vision_model.model
                trainer.model.config.use_backbone_lora = 0
            # trainer.model.merge_and_unload()
            trainer.save_model(os.path.join(args.log_dir, "ckpt_model"))
            
        print("\n" + "="*60)
        print("Training completed. Starting comprehensive evaluation...")
        print("="*60)
        evaluate_all_datasets(trainer, args, tokenizer)


if __name__ == "__main__":
    main(sys.argv[1:])