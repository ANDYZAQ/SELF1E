import torch
from utils_internvl.dataset import ValDataset

def get_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": "auto",
                "betas": [args.beta1, args.beta2],
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,  # 减小bucket size避免超时
            "allgather_bucket_size": 5e8,  # 减小bucket size避免超时
            # "offload_optimizer": {
            #     "device": "cpu"
            # },
            # "offload_param": {
            #     "device": "cpu"
            # },
        },
    }
    return ds_config

def find_linear_layers_qwen(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        "visual",
                        "lm_head",
                        "text_hidden_fcs"
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def get_all_eval_datasets():
    """Get all evaluation datasets from eval_all.sh"""
    return [
        "ReasonSeg|val",
        "ReasonSeg|test",
        "refcoco|unc|val",
        "refcoco|unc|testA", 
        "refcoco|unc|testB",
        "refcoco+|unc|val",
        "refcoco+|unc|testA",
        "refcoco+|unc|testB", 
        "refcocog|umd|test",
        "refcocog|umd|val",
        "grefcoco|unc|val",
        "grefcoco|unc|testA",
        "grefcoco|unc|testB"
    ]


def evaluate_all_datasets(trainer, args, tokenizer):
    """Evaluate model on all datasets using existing evaluate function"""
    all_datasets = get_all_eval_datasets()
    all_results = {}
    
    # Store original val_dataset to restore later
    original_val_dataset = args.val_dataset if hasattr(args, 'val_dataset') else None
    
    for dataset_name in all_datasets:
        print(f"\n--- Evaluating on {dataset_name} ---")
        args.val_dataset = dataset_name
        
        # Create validation dataset for current dataset
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            dataset_name,
            args.image_size,
            use_high_res=args.high_res,
        )
        trainer.eval_dataset = val_dataset
        
        # Run evaluation using existing function - it will save results automatically
        metrics = trainer.evaluate(metric_key_prefix="eval")
    
    # Restore original val_dataset
    if original_val_dataset is not None:
        args.val_dataset = original_val_dataset
    
    return all_results

