### Trains a classifier-based guidance on the latent space of a diffusion model
from functools import partial
import json
import os
import sys
from torch import nn
import torch
import pandas as pd
from torch.utils.data import DataLoader

from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertPooler
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from src.modeling.diffusion.gaussian_diffusion import GaussianDiffusion

from src.train_infer.factory_methods import create_model_and_diffusion
from src.utils import dist_util
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.utils.eval_smiles import calc_qed, calc_sas
from rdkit import Chem
#from src.utils.data_utils_sentencepiece import SMILESDataset
from src.utils.custom_tokenizer import create_tokenizer
from src.utils.logger import log

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class DiffusionBertForPropertyPrediction(nn.Module):
	"""A bert based model that uses the latent space of a diffusion model as input"""
	
	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", "word_embeddings.weight"]
	
	def __init__(self, config: BertConfig, diffusion_model: GaussianDiffusion, num_labels: int):
		super().__init__()
		self.diffusion_model = diffusion_model
		self.pred_model = BertModel(config)
		self.word_embeddings = nn.Embedding(
			num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
		)
		
		self.pooler = BertPooler(config)
		
		self.num_labels = num_labels
		
		self.up_proj = nn.Sequential(
			nn.Linear(config.embedding_dim, config.embedding_dim * 4),
			nn.Tanh(),
			nn.Linear(config.embedding_dim * 4, config.hidden_size),
		)
		
		self.train_diffusion_steps = config.train_diffusion_steps
		
		# self.time_embeddings = nn.Embedding(self.train_diffusion_steps + 1, config.hidden_size)
		self.time_embeddings = nn.Embedding(2000 + 1, config.hidden_size)
		
		# Model parallel
		self.model_parallel = False
		self.device_map = None
		
		self.classification_head = nn.Sequential(
			nn.Linear(config.hidden_size, config.hidden_size),
			nn.Tanh(),
			nn.Linear(config.hidden_size, self.num_labels),
		)
	
	def forward(
			self,
			input_ids=None,
			past_key_values=None,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			labels=None,
			use_cache=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
	):
		
		if inputs_embeds is None:
			# The classifier is supposed to be used with a diffusion model. During training, the embeddings should be provided
			# by the backbone model. The word_embeddings are included here for to test the classifier on its own.
			inputs_embeds = self.word_embeddings(input_ids)
		
		t = torch.randint(-1, self.train_diffusion_steps, (inputs_embeds.shape[0],)).to(
			inputs_embeds.device
		)
		# done this way because torch randint is [inclusive, exclusive), and we don't want to pass samples with t = num_diffusion_steps
		# TODO: double-check this
		t_mask = t >= 0
		
		inputs_with_added_noise = self.diffusion_model.q_sample(x_start=inputs_embeds, t=t)
		# replace the embeddings with the noisy versions for all samples with t >= 0
		inputs_embeds[t_mask] = inputs_with_added_noise[t_mask]
		inputs_embeds = self.up_proj(inputs_embeds)
		
		# essentially, t = -1 is the last step
		t[~t_mask] = self.train_diffusion_steps
		time_embedded = self.time_embeddings(t).unsqueeze(1)
		
		inputs_embeds = torch.cat([inputs_embeds, time_embedded], dim=1)
		
		outputs = self.pred_model(
			inputs_embeds=inputs_embeds,
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		
		return self.loss_from_outputs(outputs, labels)
	
	def label_logp(self, inputs_with_added_noise, t, labels):
		"""
		Returns p(labels | x_t, t) for a batch of samples. Note that inputs_with_added_noise are supposed to be the noisy versions of the inputs. Using DDPM terminology, this is the x_t.
		"""
		
		inputs_with_added_noise = self.up_proj(inputs_with_added_noise)
		
		time_embedded = self.time_embeddings(t).unsqueeze(1)
		inputs_embeds = torch.cat([inputs_with_added_noise, time_embedded], dim=1)
		outputs = self.pred_model(
			inputs_embeds=inputs_embeds,
		)
		return self.loss_from_outputs(outputs, labels)
	
	def loss_from_outputs(self, outputs, labels=None):
		pooled_output = self.pooler(outputs[0])
		logits = self.classification_head(pooled_output)
		if labels is not None:
			loss = nn.MSELoss()(logits.view(-1, self.num_labels), labels.unsqueeze(1))
		else:
			loss = None
		
		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
	
	# TODO: make num_labels a property of the config
	@staticmethod
	def load_from_checkpoint(
			checkpoint_path: str,
			config: BertConfig,
			diffusion_model: GaussianDiffusion,
			num_labels: int = 2,
	):
		model = DiffusionBertForPropertyPrediction(config, diffusion_model, num_labels)
		model.load_state_dict(torch.load(checkpoint_path), strict=False)
		return model


class StubDiffusionModel(nn.Module):
	def __init__(self):
		super().__init__()
	
	def q_sample(self, x_start, t):
		return x_start


def train_classifier_on_diffusion_latents():
	## TODO: modify and debug
	
	# Step 1: load the arguments
	args = get_training_args()
	
	# Step 2: load the model and diffusion
	model, diffusion = create_model_and_diffusion(
		**args_to_dict(args, model_and_diffusion_defaults().keys())
	)
	model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
	
	tokenizer = create_tokenizer(
		return_pretokenized=args.use_pretrained_embeddings, path=f"data/{args.dataset}/"
	)
	
	# Step 3: load the data
	dataloader = get_dataloader(
		path=f"data/{args.dataset}/{args.dataset}_labeled.tsv", tokenizer=tokenizer,
		max_seq_len=args.sequence_len
	
	)
	
	# Step 4: create the classifier
	config = BertConfig.from_pretrained("bert-base-uncased")
	config.train_diffusion_steps = args.diffusion_steps
	config.embedding_dim = args.in_channel
	config.vocab_size = tokenizer.vocab_size
	
	model = DiffusionBertForSequenceClassification(
		config=config, num_labels=2, diffusion_model=diffusion
	).to(device)
	
	# Step 5: train the classifier
	
	model = training_loop(model=model, dataloader=dataloader, num_epochs=args.classifier_num_epochs)
	
	# Step 6: save the model
	torch.save(model.state_dict(), f"{args.checkpoint_path}/classifier.pt")


def unit_test(train_epochs=50):
	#
	sentences = ['OCC1=COC(CC2OC2C#N)=N1', 'OCC1=CON=C1CC1OC1C#N', 'OCC1=COC2=C1C(NC2)C(O)=O', 'OCC1=COC2=C1CNC2C(O)=O',
				 'OCC1=C(O)N=C2COCCCN12', 'OCC1C(O)CC2N1C1=CNC=C21', 'OCC1=C(OC=O)N=CC2=C1OC2',
				 'OCC1=C(OC=O)C=NC2=C1CO2',
				 'OCC1=C(OC=O)C=NC2=C1OC2', 'OCC1=COC2=C1CN1CC(O)C21', 'OCC1=COC2=C1C1C(O)CN1C2',
				 'OCC1C(O)C(C#N)C2=CC=CN12',
				 'OCC1C(O)C(C#N)N2C=CC=C12', 'OCC1C(O)C2=C3CCOC3=CN12', 'OCC1C(O)C2=CN3CCOC3=C12',
				 'OCC1C(O)C2=C3OCCC3=CN12',
				 'OCC1C(O)C2=C3OCCN3C=C12', 'OCC1C(O)C2=CC3=C(OCC3)N12', 'OCC1COC2=C3CC(O)C3=CN12',
				 'OCC1COC2=C1N1CC(O)C1=C2']
	
	# set random seed
	import random
	
	random.seed(0)
	
	# calculate properties
	mols = [Chem.MolFromSmiles(mol) for mol in sentences if Chem.MolFromSmiles(mol) is not None]
	qeds = [calc_qed(mol) for mol in mols]
	sass = [calc_sas(mol) for mol in mols]
	
	data = pd.DataFrame({"sentence": sentences, "qed": qeds, 'sas': sass})
	
	tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
	
	data = data.sample(frac=1).reset_index(drop=True)
	sentences = data["sentence"].tolist()
	labels = data["qed"].tolist()  # debug: test with only target
	
	input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")[
		"input_ids"
	]
	
	config = BertConfig.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
	config.train_diffusion_steps = 2000
	config.embedding_dim = 128
	
	model = DiffusionBertForPropertyPrediction(
		config=config, num_labels=1, diffusion_model=StubDiffusionModel()
	).to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	count = 0
	for epoch_idx in range(train_epochs):
		batch_size = 32
		epoch_loss = 0.0
		num_batches = 0
		count += 1
		for i in range(0, len(input_ids), batch_size):
			optimizer.zero_grad()
			batch_input_ids = input_ids[i: i + batch_size]
			batch_labels = torch.tensor(labels[i: i + batch_size])
			outputs = model(input_ids=batch_input_ids.to(device), labels=batch_labels.to(device))
			outputs.loss.backward()
			optimizer.step()
			epoch_loss += outputs.loss.item()
			num_batches += 1
		
		print(f"Epoch {epoch_idx}: {epoch_loss / num_batches:.5f}")
	
	# test on new data
	test_sentences = [
		"OCC1=C(O)NC=C2NC(=O)C=C12",
		"OCC1=C(O)C=C2C(=O)NC=C2N1",
		"OC(C1COC(=O)C1)C1=NC=CN1",
		"OCC1COCCCCCN1C=O",
	]
	
	for sentence in test_sentences:
		inferred_label = get_label_from_sentence(
			sentence=sentence, model=model, tokenizer=tokenizer
		)
		gt = calc_qed(Chem.MolFromSmiles(sentence))
		print(f"{sentence} -> pred: {inferred_label}, gt: {gt}")


def unit_test_moses_qed(train_epochs=10000):

	df_chem = pd.read_csv('../../data/moses2/moses2.csv')
	df_train = df_chem[df_chem['SPLIT'] == 'train']
	df_valid = df_chem[df_chem['SPLIT'] == 'test_scaffolds']
	
	tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
	
	smiles = df_train["SMILES"].tolist()
	labels = df_train["qed"].tolist()  # debug: test with only target
	
	input_ids = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")[
		"input_ids"
	]
	
	config = BertConfig.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
	config.train_diffusion_steps = 2000
	config.embedding_dim = 128
	
	model = DiffusionBertForPropertyPrediction(
		config=config, num_labels=1, diffusion_model=StubDiffusionModel()
	).to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	
	for epoch_idx in range(train_epochs):
		batch_size = 256
		epoch_loss = 0.0
		num_batches = 0
		for i in range(0, len(input_ids), batch_size):
			optimizer.zero_grad()
			batch_input_ids = input_ids[i: i + batch_size]
			batch_labels = torch.tensor(labels[i: i + batch_size])
			outputs = model(input_ids=batch_input_ids.to(device), labels=batch_labels.to(device))
			outputs.loss.backward()
			optimizer.step()
			epoch_loss += outputs.loss.item()
			num_batches += 1
			print(f"Epoch {i}: {epoch_loss / num_batches:.5f}")
	torch.save(model.state_dict(), "qed_pred.pt")
	# test on new data
	
	# for i, sentence in enumerate(df_valid['SMILES'].tolist()):
	# 	inferred_label = get_label_from_sentence(
	# 		sentence=sentence, model=model, tokenizer=tokenizer
	# 	)
	# 	gt = df_valid['qed'].tolist()[i]
	# 	print(f"{sentence} -> pred: {inferred_label}, gt: {gt}")
		
def get_label_from_sentence(model, sentence, tokenizer):
	ids = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")["input_ids"]
	return model(ids.to(device)).logits.item()


if __name__ == "__main__":
	import sys
	#unit_test()
	unit_test_moses_qed(1)
	# if sys.argv[1] == "run_unit_tests":
	# 	unit_test_for_smiles_classification()
	# elif sys.argv[1] == "run_scaffold_tests":
	# 	unit_test_for_scaffold_classification()
	# else:
	# 	train_classifier_on_diffusion_latents()