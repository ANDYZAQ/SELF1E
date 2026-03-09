from transformers import Trainer, TrainingArguments
from utils_global.utils_global import get_ds_config
import transformers
import torch.distributed as dist


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.log_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        weight_decay=0.0,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        num_train_epochs=args.epochs,
        max_steps=args.steps_per_epoch * args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch", 
        save_steps=1,
        save_total_limit=1,
        eval_strategy="epoch" if not args.no_eval else "no",
        eval_steps=args.steps_per_epoch if not args.no_eval else None,
        load_best_model_at_end=False,
        metric_for_best_model="eval_giou" if not args.no_eval else None,
        greater_is_better=True,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        deepspeed=get_ds_config(args),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.workers,
        report_to=["tensorboard"],
    )

    return training_args

def get_training_args_seq(args):
    training_args = TrainingArguments(
        output_dir=args.log_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        weight_decay=0.0,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        num_train_epochs=args.epochs,
        # max_steps=args.steps_per_epoch * args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="steps", 
        save_steps=200,
        save_total_limit=1,
        eval_strategy="steps" if not args.no_eval else "no",
        eval_steps=200,
        load_best_model_at_end=False,
        metric_for_best_model="eval_giou" if not args.no_eval else None,
        greater_is_better=True,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        deepspeed=get_ds_config(args),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.workers,
        report_to=["tensorboard"],
    )

    return training_args