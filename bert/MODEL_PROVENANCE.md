# Model provenance

The experiment initializes both the tokenizer and classifier backbone with:

```python
BertTokenizerFast.from_pretrained("bert-base-uncased")
BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
```

No fine-tuned checkpoint or model weight is included in this repository. On an
execution machine, Transformers downloads the standard public
`bert-base-uncased` checkpoint into that machine's Hugging Face cache. The
binary classification head is newly initialized and trained separately for
each source dataset with seed 42.

The formal BERT run metadata, including the resolved Transformers and PyTorch
versions, model identifier, model-config revision, and training configuration,
is provided in `reference_results/bert/run_metadata.json`. No pretrained or
fine-tuned weights are redistributed.
