import os
import torch
from docx import Document
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Global lists to store requirements and offers
requirements = []
offers = []

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set pad token to eos token for GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Function to read .docx files and extract text
def read_docx(folder_path):
    files = os.listdir(folder_path)
    docx_files = [f for f in files if f.endswith('.docx')]
    target_text_for_split = "Oferta pentru"

    for docx_file in docx_files:
        file_path = os.path.join(folder_path, docx_file)
        document = Document(file_path)
        all_paragraphs = [paragraph.text for paragraph in document.paragraphs]
        full_text = '\n'.join(all_paragraphs)

        if target_text_for_split in full_text:
            parts = full_text.split(target_text_for_split)
            if len(parts) > 1:
                requirements.append(parts[0].strip())
                offer_part = parts[1].strip()
                if "Scopul documentului" in offer_part:
                    offers.append(offer_part.split("Scopul documentului")[1].strip())
                else:
                    offers.append(offer_part)

# Function to preprocess data
def preprocess_data(requirements, offers):
    return [{'text': req + '\n' + offer} for req, offer in zip(requirements, offers)]

# Read and preprocess data
read_docx("../Oferte test")
data = preprocess_data(requirements, offers)
dataset = Dataset.from_list(data)
"""""
# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Tokenize and prepare datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
def split_dataset(dataset, split_ratio=0.1):
    split = int(len(dataset) * split_ratio)
    return dataset.select(range(split)), dataset.select(range(split, len(dataset)))

train_dataset, eval_dataset = split_dataset(tokenized_datasets)

# Collate function for training
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids  # Use input_ids as labels for language modeling
    }

# TrainingArguments configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer with both train and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("fine-tuned-gpt2")
tokenizer.save_pretrained("fine-tuned-gpt2")
"""
model = GPT2LMHeadModel.from_pretrained("fine-tuned-gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("fine-tuned-gpt2")


def generate_text(prompt, max_length=150, num_return_sequences=1):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate text
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,  # Set to a higher value to ensure enough length
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,  # Adjust for more creativity
        top_k=50,  # Adjust for diversity
        top_p=0.9  # Adjust for nucleus sampling
    )


# Example prompt
prompt = "O aplicatie de tip livrare de mancare"

# Generate text
generated_text = generate_text(prompt)

# Print the generated text
print(generated_text)



