#!pip install datasets
#!pip install evaluate
#!pip install seqeval
#!pip install accelerate -U
#!pip install -U transformers
#!pip install sklearn

"""Script created with help from the following tutorial:
    https://huggingface.co/docs/transformers/tasks/token_classification#inference
"""

import json
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Check if the number of arguments is correct
if len(sys.argv) < 4:
    print("Usage: python script_name.py arg1 arg2 arg3 {only_positive}")
else:
    # Retrieve the arguments
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    base_path = sys.argv[3]
    only_positives = False if len(sys.argv) < 5 else True

    print(only_positives)
    
    # Print the arguments
    print("Running for model", model_name)
    print("Loading dataset", dataset_name)
    print("Expecting to retrieve and store data from:", base_path)
    print('Running with only positive instances:', only_positives)

data = load_dataset(dataset_name)
if only_positives:
    data = data.filter(lambda item: item['label'] == 1)
data = data.remove_columns(['text', 'label'])

def make_lower(item):
    item['tokens'] = [text.lower() for text in item['tokens']]
    return item
data = data.map(make_lower)

print('Initializing Tokenizer...')
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

example = data['train'][0]

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

from transformers import DataCollatorForTokenClassification

print('Tokenizing and aligning labels...')

tokenized_data = data.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

import numpy as np
import evaluate

seqeval = evaluate.load("seqeval")

label_list = ["Non-Pun", "Pun"]

labels = [label_list[i] for i in example[f"labels"]]

def compute_metrics(p, convert_predictions=True):
    predictions, labels = p
    if(convert_predictions):
        predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

id2label = {
    0: "Non-Pun",
    1: "Pun"
}
label2id = {
    "Non-Pun": 0,
    "Pun": 1
}

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

print("Starting training...")

from transformers import TrainingArguments, Trainer

output_path = Path(base_path)
output_path.mkdir(exist_ok=True, parents=True)
output_dir = output_path / model_name.split('/')[1]
print(f'output_dir: {output_dir}')
 
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.02,
    evaluation_strategy="steps",  
    eval_steps=500,               
    save_strategy="steps",        
    save_steps=500,               
    load_best_model_at_end=True,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    gradient_accumulation_steps=2,
    push_to_hub=False,
    hub_private_repo=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],  
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print('Training done, evaluating the model...')
# Evaluate the model
results = trainer.evaluate(eval_dataset=tokenized_data["test"])

with open(f"{base_path}\evaluation_results.json", "w") as file:
    file.write(str(results))

"""Post-Processing"""
import torch
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification

best_path = [model for model in output_dir.iterdir()][0]
tokenizer = AutoTokenizer.from_pretrained(best_path, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForTokenClassification.from_pretrained(best_path)

def extract_tokens(tokenized_data, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokenized_data['input_ids'])
    return tokens

def predict_pun_location(entry, model, post_process_func=None):
    with torch.no_grad():
        # Ensure input IDs are converted to a tensor and unsqueezed to add a batch dimension
        input_ids = torch.tensor(entry['input_ids'], dtype=torch.long).unsqueeze(0)
        # Generate outputs using the model
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        # Obtain the most likely token class indices
        predictions = torch.argmax(logits, dim=2)

        # Apply post-processing if a function is provided
        if post_process_func:
            predictions[0] = post_process_func(predictions[0], entry['labels'])

        return predictions

def post_processing_first_word(predictions, labels):
    inverted_predictions = predictions.flip(0)
    inverted_labels = list(reversed(labels))
    found_pun = False
    pun_index = -1

    for i in range(len(inverted_predictions)):
        if inverted_labels[i] == -100:
            continue
        if inverted_predictions[i] == 1 and not found_pun:
            found_pun = True
            pun_index = i
        if found_pun and i != pun_index:
            inverted_predictions[i] = 0

    return inverted_predictions.flip(0)

def post_processing_seq_first_word(predictions, labels):
    inverted_predictions = predictions.flip(0)
    inverted_labels = list(reversed(labels))
    found_start_sequence = False
    ended_sequence = False
    pun_index = -1

    for i in range(len(inverted_predictions)):
        if inverted_labels[i] == -100:
            continue
        elif inverted_predictions[i] == 1 and not found_start_sequence:
            found_start_sequence = True
            pun_index = i
        elif found_start_sequence and inverted_predictions[i] == 0:
          ended_sequence = True
        elif ended_sequence:
            inverted_predictions[i] = 0

    return inverted_predictions.flip(0)

def process_predictions_for_eval(tokenized_data,  model, post_processing_first_word = None):
  all_predictions = []
  for entry in tokenized_data:
    prediction = predict_pun_location(entry, model, post_processing_first_word)
    all_predictions.append(prediction.squeeze().tolist())

  max_len = max(len(pred) for pred in all_predictions)

  all_predictions_padded = [pred + [-100] * (max_len - len(pred)) for pred in all_predictions]

  predicted_indices = np.array(all_predictions_padded)

  return predicted_indices

initial_predictions = process_predictions_for_eval(tokenized_data['test'], model)

post_processing_1 =  process_predictions_for_eval(tokenized_data['test'], model, post_processing_first_word)
post_processing_results_1 = compute_metrics((post_processing_1, tokenized_data['test']['labels']), False)
with open(f"{base_path}\postproc_firstonly_results.json", "w") as file:
    json.dump(post_processing_results_1, file, indent=4)

post_processing_2 =  process_predictions_for_eval(tokenized_data['test'], model, post_processing_seq_first_word)
post_processing_results_2 = compute_metrics((post_processing_2, tokenized_data['test']['labels']), False)
with open(f"{base_path}\postproc_sqd_results.json", "w") as file:
    json.dump(post_processing_results_2, file, indent=4)

"""Pun detection"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

"""Reload the dataset with all the examples"""
pun_dec_data = load_dataset(dataset_name)
pun_dec_data = pun_dec_data.map(make_lower)
dec_tokenized_data = pun_dec_data.map(tokenize_and_align_labels, batched=True)

def evaluate_binary_classification(predictions, actuals):
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions, output_dict=True)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "classification_report": report
    }

    return results

pun_location_predictions = []

for entry in dec_tokenized_data:
    entry['pred_loc'] = predict_pun_location(entry, model, False)
    pun_location_predictions.append(entry)

for entry in pun_location_predictions:
    entry['pred_det'] = 0
    pred_loc_list = entry['pred_loc'].squeeze().tolist()

    for i in range(len(entry['labels'])):
        label = entry['labels'][i]
        if label != -100:
            if pred_loc_list[i] == 1:
                entry['pred_det'] = 1

predicted_labels = [entry['pred_det'] for entry in pun_location_predictions]
actual_labels = [entry['label'] for entry in pun_location_predictions]

results = evaluate_binary_classification(predicted_labels, actual_labels)

print(f"precision: {results['precision']}")
print(f"recall: {results['recall']}")
print(f"f1: {results['f1']}")
print(f"accuracy: {results['accuracy']}")
with open(f"{base_path}\pundec.json", "w") as file:
    json.dump(results, file, indent=4)





