
# Fine-Tuning BERT for Named Entity Recognition (NER)

This project focuses on fine-tuning a pre-trained BERT model for the Named Entity Recognition (NER) task using the CoNLL-2003 dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Inference](#inference)
- [References](#references)

## Installation

To run the notebook, you need to install the following Python libraries:

```bash
pip install transformers datasets tokenizers seqeval
```

## Dataset

The dataset used in this project is the [CoNLL-2003 dataset](https://huggingface.co/datasets/conll2003), which is loaded using the `datasets` library from Hugging Face.

## Preprocessing

The notebook includes preprocessing steps such as tokenization using `BertTokenizerFast` and aligning the NER tags with the tokenized words. This ensures that the labels correspond correctly to the tokens, including subword tokens.

## Model Training

The fine-tuning process involves training the `BertForTokenClassification` model on the tokenized dataset. The training is handled using the `Trainer` class from the `transformers` library, which manages the training loop, evaluation, and logging.

## Evaluation

Evaluation metrics such as precision, recall, F1 score, and accuracy are calculated to assess the model's performance. The `seqeval` library is used for computing these metrics.

## Saving the Model

The fine-tuned model and the tokenizer are saved locally for future use. The label mappings (`id2label` and `label2id`) are also updated in the model's configuration file.

## Inference

An example of how to use the fine-tuned model for NER inference is provided in the notebook. A sample sentence is processed, and the NER results are displayed.

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)
