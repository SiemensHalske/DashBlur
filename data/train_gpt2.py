from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import GPT2Config, GPT2Tokenizer
from transformers.trainer_callback import TrainerCallback
from transformers import GPT2Tokenizer

import requests
import ssl
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl._create_unverified_context()
        kwargs["ssl_context"] = context
        return super(NoSSLAdapter, self).init_poolmanager(*args, **kwargs)


class PrintCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        print(
            f"Epoch: {state.epoch} Iteration: {state.global_step} Loss: {state.loss}")


session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = NoSSLAdapter(max_retries=retry)
session.mount('https://', adapter)

# Initialize tokenizer with no ssl
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", cache_dir="./cache", use_fast=True, session=session)

# Create a text dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_dataset.txt",
    block_size=128,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Create an evaluation text dataset
eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="eval_dataset.txt",
    block_size=128,
)

# Initialize GPT-2 configuration and model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_ctx=128,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-chatbot",
    overwrite_output_dir=True,
    num_train_epochs=32,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    callbacks=[PrintCallback()],
)

# Train the model
trainer.train()
trainer.evaluate()
