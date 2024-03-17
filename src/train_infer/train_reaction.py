"""
Train a diffusion model on molecules
"""

import json, os
import pathlib
import pprint
import sys
import wandb
from transformers import set_seed
import os
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from src.utils import dist_util, logger
from src.modeling.diffusion.resample import create_named_schedule_sampler
from factory_methods import create_model_and_diffusion
from train_loop import TrainLoop
from src.utils import data_utils_sentencepiece
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.utils.custom_tokenizer import create_tokenizer


def main():
	args = create_argparser().parse_args()
	set_seed(args.seed)
	dist_util.setup_dist()
	logger.configure()
	pathlib.Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
	
	### load data
	# if no eval data, will random split the train data into train set and valid set.
	
	logger.log("loading data and creating data loader")
	tokenizer = AutoTokenizer.from_pretrained(args.config_name)  # default charTokenizer.
	#tokenizer.save_pretrained("./char_tokenizer")
	args.vocab_size = tokenizer.vocab_size
	
	# dataset name: gdb13 or moses2 or guacamol2
	if args.dataset == 'patents':
		df_train = pd.read_csv('./data/patents/Schwaller_US_patents_1976_Sep2016_train_dataset_cleaned_HX.csv')
		df_valid = pd.read_csv('./data/patents/Schwaller_US_patents_1976_Sep2016_valid_dataset_cleaned_HX.csv')
		df_test = pd.read_csv('./data/patents/Schwaller_US_patents_1976_Sep2016_test_dataset_cleaned_HX.csv')

	else:
		assert False, "Invalid dataset"
		
	train_dataloader = data_utils_sentencepiece.get_dataloader(df_train,
															   tokenizer,
															   batch_size=args.batch_size,
															   max_length=args.sequence_len,
															   shuffle=True,
															   reaction_data=True)
	val_dataloader = data_utils_sentencepiece.get_dataloader(df_valid,
															   tokenizer,
															   batch_size=args.batch_size,
															   max_length=args.sequence_len,
															   shuffle=False,
															   reaction_data=True)
	### create diffusion model
	
	logger.log("creating model and diffusion...")
	
	model, diffusion = create_model_and_diffusion(
		**args_to_dict(args, model_and_diffusion_defaults().keys())
	)
	model.to(dist_util.dev())
	
	print(model)
	
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	
	logger.log(f"the parameter count is {pytorch_total_params}")
	schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
	
	logger.log(f"saving the hyperparameters to {args.checkpoint_path}/training_args.json")
	with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
		json.dump(args.__dict__, f, indent=2)
		
	### start training
	# TODO: (optional) set wandb for training visualization
	
	#    if args.debug:
	#        wandb.init(mode="disabled")
	#    else:
	#        wandb.init(
	#            project=os.getenv("WANDB_PROJECT", "minimial-text-diffusion"),
	#            name=args.checkpoint_path + make_wandb_name_from_args(args),
	#            notes=args.notes,
	#        )
	#        wandb.config.update(args.__dict__, allow_val_change=True)
	
	logger.log("training...")
	TrainLoop(
		model=model,
		diffusion=diffusion,
		data=iter(train_dataloader),
		batch_size=args.batch_size,
		microbatch=args.microbatch,
		lr=args.lr,
		ema_rate=args.ema_rate,
		log_interval=args.log_interval,
		save_interval=args.save_interval,
		resume_checkpoint=args.resume_checkpoint,
		use_fp16=args.use_fp16,
		fp16_scale_growth=args.fp16_scale_growth,
		schedule_sampler=schedule_sampler,
		weight_decay=args.weight_decay,
		lr_anneal_steps=args.lr_anneal_steps,
		checkpoint_path=args.checkpoint_path,
		gradient_clipping=args.gradient_clipping,
		eval_data=iter(val_dataloader),
		eval_interval=args.eval_interval,
	).run_loop()

# def make_wandb_name_from_args(args):
#     keys_to_add = ["batch_size", "lr", "num_heads", "lr_anneal_steps", "config_name", "seed", "in_channel"]
#     name = ""
#     for key in keys_to_add:
#         name += f"{key}={getattr(args, key)}_"
#     return name


if __name__ == "__main__":
	main()