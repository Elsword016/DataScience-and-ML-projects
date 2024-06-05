## Leash Bio - Predict New Medicines with BELKA

This is an ongoing Kaggle Competition - to develop machine learning (ML) models to predict the binding affinity of small molecules to specific protein targets – a critical step in drug development for the pharmaceutical industry that would pave the way for more accurate drug discovery. You’ll help predict which drug-like small molecules (chemicals) will bind to three possible protein targets.

Competition info and datasets: [Leash Bio - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)

## Approaches:

The dataset is massive, with lots of data points. Due to computing restrictions, I sampled 30000 points from the dataset while ensuring there was no class imbalance. I used two approaches: 
  1. Calculating fingerprints from SMILES and then using Xgboost to predict binding.
  2. Finetuning a single BERT model by concatenating inputs of SMILES strings and protein names using a `[SEP]` token.

For finetuning the BERT model, data was split into 80-20, where 80% was for training and 20% for validation.

### Simple XGBoost Classifier(`belka_xgboost.ipynb`)

The `RdKit` library is used to compute molecular fingerprints (Ecfp) from the SMILES structures, which are used as features to predict the target binding. For evaluation, I calculated the mAP and accuracy scores using the `sklearn` library.

**Confusion Matrix**

![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/a1fad7f4-e95a-47eb-9de3-dbb3fb3e198b)

**Eval Metrics**

MAp score: 0.96, Accuracy: 0.898

Seems like Xgboost performed quite well even without hyperparameter adjustments.

### Prediction with BERT

I used the HuggingFace🤗 transformers library for the models and tokenizers. Checkpoint used: `bert-base-uncased`. For finetuning the model, the I concatenated the SMILES and the protein names using the special separator `[SEP]` token. The `[SEP]` token is commonly used in BERT to separate different segments of texts in tasks like question answering where the input consists of two distinct parts (e.g., a question and a context).

For example:

```python
smiles = ''Cc1conc1CNc1nc(Nc2cccnc2C)nc(N[C@H](CC(=O)N[Dy])c2ccc(Cl)cc2)n1'
protein_name = 'BRD4'
input_text = f{smiles} [SEP] {protein_name}
#input text
'Cc1conc1CNc1nc(Nc2cccnc2C)nc(N[C@H](CC(=O)N[Dy])c2ccc(Cl)cc2)n1[SEP]BRD4'
# tokenize this input_text
```
The [SEP] token is crucial as it separates the two text inputs. This allows the model to understand that it is processing two related but distinct segments of data.

After that just fine-tune the model like a text-classification task. I used HuggingFace `Trainer` for this, but can be done in pure `PyTorch` or `PyTorch Lightning`.


**Training Configs**

Metrics used: **accuracy** from `evaluate` library

```python
training_args = TrainingArguments(
    output_dir='Belka-BERT',
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='logs',
    report_to='wandb',
    learning_rate=2e-5,
)
```
**Training Results**

Room for more improvement, tuning the hyperparams, and more training epochs might be needed.

```python
{'eval_loss': 0.3811909556388855, 'eval_accuracy': 0.8923333333333333, 'eval_runtime': 9.606, 'eval_samples_per_second': 624.61, 'eval_steps_per_second': 78.076, 'epoch': 4.0}
{'train_runtime': 871.4642, 'train_samples_per_second': 110.159, 'train_steps_per_second': 13.77, 'train_loss': 0.34040866724650065, 'epoch': 4.0}
```
The results were logged with the `Weights and Biases (Wandb)` dashboard.

### Gradio Interface

For a faster prediction, I built a small `gradio` interface to interact with the fine-tuned model. The interface accepts the SMILES of the molecule and the name of the protein and then outputs the Prediction (Bind/No bind) and the scores (`logits.softmax(dim=1).max().item()`)

![image](https://github.com/Elsword016/DataScience-and-ML-projects/assets/29883365/53848cbd-5ff4-426f-bfe0-13b427c39a00)





