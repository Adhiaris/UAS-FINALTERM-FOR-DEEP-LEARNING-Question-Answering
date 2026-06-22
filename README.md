# Task 2: Question Answering with T5-base on SQuAD

## Purpose

This repository contains the implementation of generative question answering using T5-base fine-tuned on the SQuAD v1.1 dataset. The goal is to demonstrate how an encoder-decoder (Seq2Seq) model can be trained to generate free-text answers given a question and a context passage.

## Project Overview

Unlike BERT-style extractive QA that predicts start and end token positions within a context, T5 treats question answering as a text-to-text generation problem. The model receives a formatted input and generates the answer token by token.

Input format:
```
question: {question text}  context: {context passage}
```

Output:
```
{generated answer}
```

The notebook covers dataset exploration, tokenization using T5's text-to-text format, Seq2Seq fine-tuning, evaluation using Exact Match and Token F1, inference demo, and model saving.

## Models Used

| Model | Architecture | Parameters |
|-------|-------------|------------|
| t5-base | Encoder-Decoder Transformer | 220M |

T5 (Text-to-Text Transfer Transformer) frames every NLP task as a text generation problem. The encoder processes the question and context, while the decoder auto-regressively generates the answer. Beam search with `num_beams=4` is used during inference for improved answer quality.

## Dataset

| Split | Samples Used | Total Available |
|-------|-------------|-----------------|
| Train | 15,000 | 87,599 |
| Validation | 1,000 | 10,570 |

Dataset: `rajpurkar/squad` (SQuAD v1.1)

## Results

| Metric | Value |
|--------|-------|
| Exact Match (EM) | 55-65% |
| Token F1 | 65-75% |
| Eval Loss | -- |
| Train Samples | 15,000 |
| Eval Samples | 1,000 |

Note: T5 generative QA scores lower than extractive BERT-style models on SQuAD because SQuAD answers are spans directly extracted from the context, which favors extractive approaches.

| Approach | Model | Method | Typical EM |
|----------|-------|--------|-----------|
| Extractive | BERT-base | Span prediction | ~80% |
| Generative | T5-base | Text generation | ~55-65% |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Learning Rate | 3e-4 |
| Batch Size (train) | 16 |
| Batch Size (eval) | 32 |
| Max Input Length | 512 tokens |
| Max Target Length | 64 tokens |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Num Beams (inference) | 4 |
| Mixed Precision | fp16 (GPU) |

## How to Navigate

```
Task2_T5_QA_SQuAD_FIXED.ipynb
|
|-- Step 1: Install Dependencies
|-- Step 2: Load & Explore SQuAD
|-- Step 3: Tokenization & Preprocessing
|   |-- T5 text-to-text input format
|   |-- Subset selection (15k train / 1k eval)
|
|-- Step 4: Load T5-base Model
|-- Step 5: Training
|   |-- Exact Match and Token F1 metric functions
|   |-- Seq2SeqTrainingArguments
|   |-- Seq2SeqTrainer with EarlyStoppingCallback
|   |-- Training loop
|
|-- Step 6: Evaluation
|   |-- Trainer evaluation
|   |-- Manual generation on 200 validation samples
|   |-- Visualizations (F1 distribution, EM pie chart, training curves)
|
|-- Step 7: Inference Demo
|   |-- SQuAD validation samples
|   |-- Custom context QA
|
|-- Step 8: Save Model & Metrics
```

## Requirements

```
transformers
datasets
evaluate
accelerate
sentencepiece
rouge_score
torch
```

## Notes

- The preprocessing uses `text_target` and `max_target_length` parameters instead of the deprecated `as_target_tokenizer()` context manager.
- During generation, `max_length=input_length + MAX_TARGET` is used instead of `max_new_tokens` to avoid conflicts with T5's default `max_length=20`.
- The `Seq2SeqTrainer` with `predict_with_generate=True` is required for proper sequence generation during evaluation.
