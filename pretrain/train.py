import os
import torch
from itertools import chain
from datasets import load_dataset
import hydra
from typing import Optional, List,Dict
from transformers import (
    AutoConfig,
    # AutoModelForCausalLM, # Not needed if using AutoModel with custom
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, # We use our custom trainer now
    TrainingArguments,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel
)
from evaluate import load
from utils.utils import load_config, get_gpu_memory
# from torch.utils.tensorboard import SummaryWriter # Trainer handles TensorBoard
# Make sure LLaDATrainer is imported or defined before use
# from your_trainer_file import LLaDATrainer # Example import
# Or define the class directly in this script as shown above
import torch.nn.functional as F
from transformers import Trainer,DataCollatorForLanguageModeling, TrainerCallback
import math # Import math for isnan check
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import TRAINER_STATE_NAME
from rich import traceback
from typing import Dict, Tuple, Union
from .my_decoder_only_model import MyDecoderOnlyModel, CustomConfig
import logging
import wandb
from omegaconf import OmegaConf
from transformers.integrations import WandbCallback
from .my_decoder_only_model import test_model
import argparse
from torch import Tensor
from .train_diffusion import DiffusionTrainer

logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(logging.FileHandler("logs/train.log"))
# 启用彩色回溯
traceback.install()


# 设置环境变量以优化CUDA内存分配 (Optional, keep if needed)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
@hydra.main(
    config_path=".",
    config_name="config",
    version_base=None,
)
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,trust_remote_code=True,cache_dir=config.tokenizer_cache,local_files_only=True)
    AutoConfig.register("autodiffusion", CustomConfig)
    AutoModel.register(CustomConfig, MyDecoderOnlyModel)
    # Load the custom model
    customConfig = AutoConfig.from_pretrained(config.model.save_dir,trust_remote_code=True)
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.mode == "train":

        # --- Configuration ---
        output_path = "results/nightmare_1.5b" # Changed output path
        model_path = "./model" # Base for tokenizer
        custom_model_load_path = "Qwen/Qwen2.5-0.5B" # Path to your custom model config/weights
        #config_yaml_path = "./pretrain/config.yaml" # Path to your YAML config

        # 初始化 W&B
        wandb.init(
            project=config.wandb.project,  # 替换为你的项目名称
            name=config.wandb.name,         # 替换为你的运行名称           # 可选：传递你的配置
        )
        # --- Load Model and Tokenizer ---
        model = AutoModel.from_pretrained(
            config.model.custom_model_load_path,
            config=customConfig, # Pass your loaded config if needed
            trust_remote_code=True, # Be cautious with this
            local_files_only=True,
        ).to(device) 
        #test_model(model, customConfig) # Test the model with the tokenizer
        # Load your custom config dict/object
        # If self_config needs conversion to a HuggingFace Config object:
        # from transformers import PretrainedConfig
        # if isinstance(self_config, dict):
        #     # Assuming your CustomConfig can be initialized from a dict
        #     # Or map fields manually if needed
        #     hf_config = CustomConfig(**self_config)
        # else:
        #     hf_config = self_config # Assume it's already a HF compatible config object

        # Ensure CustomConfig and MyDecoderOnlyModel are registered *before* from_pretrained
        # This registration might need to be adapted based on how your CustomConfig/Model work
        # Assuming CustomConfig is a class inheriting from PretrainedConfig
        # Assuming MyDecoderOnlyModel inherits from PreTrainedModel
        # AutoModel.register(CustomConfig, MyDecoderOnlyModel) # You might need specific class names
        #print("-------------Tokenizer path:", os.path.abspath(config.tokenizer_name))
        # Load Tokenizer - Use the original base model's tokenizer usually
        
        # *** Important: Define or add the [MASK] token if it doesn't exist ***
        class SelfTrainer(Trainer):
            """
            Custom Trainer class for LLaDA-style training.
            """
            def __init__(self, *args,debug:bool=False,**kwargs):
                super().__init__(*args, **kwargs)
                self.debug = debug
                self.avg_loss = 0.0
                self._togger_loger()
            def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, Tensor], num_items_in_batch:Optional[int]=None ,return_outputs: bool = False):
                
                #logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
                #logger.info(f"Attention mask shape: {inputs['attention_mask'].shape}")
                # --- Forward Pass ---
                # logger.info(f"Model input shapes: {{k: v.shape for k, v in model_inputs.items()}}")
                outputs = model(**inputs)
                logits = outputs
                labels = inputs.get("labels")
                logger.info(f"Logits shape: {logits.shape}")
                #logger.info(f"Labels shape: {labels.shape}")

                # Shift logits and labels for autoregressive loss
                shift_logits = logits[:,:-1, :].contiguous()  # 去掉最后一个时间步
                shift_labels = labels[:, 1:].contiguous()  # 去掉第一个时间步

                # Flatten logits and labels for CrossEntropyLoss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # 返回损失
                # 计算平均损失
                self.avg_loss = loss.item()
                return (loss, outputs) if return_outputs else loss

            def compute_metrics(self, eval_pred):
                # 计算分类指标
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=-1)
                clf_results = clf_metrics.compute(predictions=preds, references=labels)
                
                # 添加 PPL
                clf_results["perplexity"] = np.exp(self.avg_loss)  
                return clf_results
            
            def _togger_loger(self ):
                if self.debug == False:
                    logger.setLevel(logging.WARNING)

            def _save_checkpoint(self, model, trial, metrics=None):
                # Save the model and tokenizer
                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.save_model(output_dir, _internal_call=True)
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                # Save the tokenizer
                logger.info(f"Tokenizer saved to {self.args.output_dir}")
        
        # Load Model - Use AutoModel for flexibility with custom architectures
        # Ensure the config used here matches the one saved in custom_model_load_path
        # Or pass your loaded hf_config explicitly
        # 注册自定义配置和模型

        #t
        # Resize token embeddings if new tokens were added
    

        # --- Load Dataset ---
        from datasets import load_from_disk

        # 指定保存的目录路径
        output_dir = config.data.output_dir  # 替换为实际保存的路径

        # 加载分词后的数据集
        logger.info(f"Loading preprocessed dataset from {output_dir}...")
        preprocessed_dataset :DatasetDict = load_from_disk(config.data.output_dir)

        # 打印数据集信息
        logger.info(f"Loaded dataset with {len(preprocessed_dataset)} {preprocessed_dataset.keys()} examples.")

        # 划分数据集为训练集和验证集
        split_dataset = preprocessed_dataset['train'].train_test_split(test_size=0.1, seed=42)

        # 获取训练集和验;证集
        #[{input:[],mask:[]},{}]
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        # 打印划分后的数据集信息
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
        # 检查 train_dataset 中样本的最大长度
        #logger.info(f"Maximum sequence length in train_dataset: {train_dataset[0]}")
        # --- Data Collator ---
        # Use the standard LM collator. It will handle padding.
        # Our custom trainer ignores the 'labels' it creates and uses 'input_ids'.
        class TruncatingDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
            def __init__(self, tokenizer, max_length, **kwargs):
                super().__init__(tokenizer, **kwargs)
                self.max_length = max_length

            def __call__(self, examples):
                # 截断 input_ids 和 attention_mask
                for example in examples:
                    example["input_ids"] = example["input_ids"][:self.max_length]
                    if "attention_mask" in example:
                        example["attention_mask"] = example["attention_mask"][:self.max_length]
                return super().__call__(examples)

        collator = TruncatingDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,  # 确保与模型的最大长度一致
            mlm=False  # 如果是 Causal LM，设置为 False
        )
        class EarlyStoppingCallback(TrainerCallback):
            def __init__(self, early_stopping_patience=3, min_delta=0.0):
                self.patience = early_stopping_patience
                self.min_delta = min_delta
                self.best_metric = None
                self.no_improvement_counter = 0

            def on_evaluate(self, args, state, control, **kwargs):
                current_metric = state.log_history[-1]['eval_loss']  # 监控验证损失
                if self.best_metric is None or current_metric < (self.best_metric - self.min_delta):
                    self.best_metric = current_metric
                    self.no_improvement_counter = 0
                else:
                    self.no_improvement_counter += 1
                    if self.no_improvement_counter >= self.patience:
                        control.should_training_stop = True  # 触发停止训练

        
        # --- Compute Metrics (Optional but good for monitoring) ---
        

        # --- Training Arguments ---
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        # Ampere GPUs (e.g., A100) support BFloat16
            bf16 = True
            fp16 = False
        else:
        # Use FP16 for older GPUs
            bf16 = False
            fp16 = True
        training_args = TrainingArguments(
            output_dir=config.training.output_path,
            overwrite_output_dir=True,
            learning_rate=1e-4, # Adjust as needed
            warmup_ratio=0.05, # Common ratio for large PT
            lr_scheduler_type="cosine",
            num_train_epochs=config.training.epochs, # Usually many more for pre-training
            per_device_train_batch_size=config.training.batch_size, # Adjust based on VRAM (12 might be too high for 14B)
            per_device_eval_batch_size=config.training.batch_size,  # Adjust based on VRAM
            gradient_accumulation_steps=16*4, # Increase to simulate larger batch size (e.g., 4*16*num_gpus = effective batch)
            gradient_checkpointing=True, # Saves VRAM at cost of ~20-30% slower training
            save_strategy="steps",
            save_steps=10000, # Save more frequently during long PT
            save_total_limit=3, # Keep last 3 checkpoints
            bf16=bf16, # Use BFloat16 if supported (Ampere GPUs+)
            fp16=fp16, # Use FP16 if BF16 not supported
            logging_strategy="steps",
            logging_steps=config.training.log_step, # Log more frequently
            eval_strategy="steps", # Evaluate periodically one 
            eval_steps=config.training.log_step, # Evaluate every 5k steps
            report_to="wandb", # Log to TensorBoard
            remove_unused_columns=False, # Important: Keep input_ids for our custom loss
            load_best_model_at_end=True,  # 训练结束时加载最佳模型
            metric_for_best_model="eval_loss", # 或 "accuracy" 等，用于确定最佳模型
            greater_is_better=False,       # 对于 loss，False 表示越小越好
            run_name="train-eval-demo-steps",
            
            # load_best_model_at_end=True, # Optional: Reload best checkpoint at the end
        )

        
        # --- Initialize Custom Trainer ---
        trainer = SelfTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            debug = config.training.debug,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                # 内置的检查点回调会自动启用[9](@ref)
            ]
            #compute_metrics=compute_metrics_simple, # Add back if you want eval metrics
            #mask_token_id=mask_token_id, # Pass the determined mask token ID
        )

        # --- Detect Last Checkpoint for Resuming ---
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint:
                print(f"Checkpoint detected, resuming training from: {last_checkpoint}")
            elif any(f.startswith("checkpoint-") for f in os.listdir(training_args.output_dir)):
                print(f"Warning: Checkpoint folders found in {training_args.output_dir}, but get_last_checkpoint failed. Specify path manually if needed.")
            else:
                print(f"No checkpoint found in {training_args.output_dir}. Starting training from scratch.")

        # --- Start Training ---
        print("Starting training...")
        try:
            # Pass resume_from_checkpoint=True to automatically use last_checkpoint if found
            # Or pass the specific path: resume_from_checkpoint=last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

            # train_result contains metrics like train_runtime, train_samples_per_second etc.
            print("Training finished.")
            print(f"Train Results: {train_result.metrics}")

            # Save final training metrics
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)

            final_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
            trainer.save_metrics("eval", final_metrics)

        except Exception as e:
            logger.error(f"An error occurred during training: ")
            import traceback
            traceback.print_exc()
            # Optionally save state even if training failed mid-way
            logger.error("Attempting to save trainer state due to error...")
            trainer.save_model() # Saves optimizer, scheduler, rng states etc. in output_dir/trainer_state.json
            trainer.save_state() # Saves model, tokenizer, and training state


        # --- Save Final Model ---
        # This saves the model state at the end of training (or after the last successful save step if interrupted)
        print("Saving final model...")
        # Ensure the final save path exists
        final_save_path = os.path.join(training_args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path) # Saves model weights, config
        tokenizer.save_pretrained(final_save_path) # Saves tokenizer files
        print(f"Final model and tokenizer saved to {final_save_path}")
    elif config.mode == "sample":  
        @torch.no_grad()
        def _inference(config):
            model = AutoModel.from_pretrained(
                config.training.final_path,
                config=customConfig, # Pass your loaded config if needed
            ).eval().to(device)
            input_text = "Please tell me some information about China "
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=config.max_seq_len)
            input_ids :Tensor = inputs.input_ids.to(device)
            past_key_values = None  # 初始化缓存

            print(input_text)
            for _ in range(config.max_seq_len - len(tokenizer.encode(input_text)) ):
                try:
                    optimized_model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                except e:
                    optimized_model = model
                outputs = optimized_model(
                    input_ids=input_ids,
                    attention_mask=inputs.attention_mask.to(device),
                )
                logger.info(f"Output shape: {outputs.shape}")
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                next_token_ids = next_token.unsqueeze(-1).to(device)
                logger.info(f"Next token: {next_token_ids.device } {input_ids.device}")
                assert next_token_ids.device == input_ids.device, "Device mismatch between input_ids and next_token_ids"
                input_ids = torch.cat([input_ids, next_token_ids], dim=-1)
                print(tokenizer.decode(next_token), end=" ", flush=True)
        
        _inference(config)
    elif config.mode == "ppl_eval":
        pass



if __name__ == "__main__":
    # This is just a placeholder. The script is designed to be run directly.
    # You can add command-line argument parsing if needed.
    main()