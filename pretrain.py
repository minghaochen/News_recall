from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, LineByLineTextDataset


tokenizer = ByteLevelBPETokenizer()


paths = './data/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(paths)

dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = './data/LineText_for_BERT.txt',
    block_size = 512  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
config = BertConfig(
    vocab_size=28198,
    hidden_size=768, 
    num_hidden_layers=12, 
    num_attention_heads=12,
    max_position_embeddings=512
)
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./pretrain_model',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_steps=10000,
    save_total_limit=5,
    prediction_loss_only=True,
#     no_cuda = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print('begin training')
trainer.train()
trainer.save_model('./pretrain_model')