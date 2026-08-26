# Model provenance

The experiment initializes both the tokenizer and classifier backbone with:

```python
BertTokenizerFast.from_pretrained("bert-base-uncased")
BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
```

No fine-tuned checkpoint or model weight is included in this package. On an
execution machine, Transformers downloads the standard public
`bert-base-uncased` checkpoint into that machine's Hugging Face cache. The
binary classification head is newly initialized and trained separately for
each source dataset with seed 42.

For the camera-ready rerun, record the resolved Transformers version and model
identifier from `results/bert_ready/run_metadata.json` and retain the server's
Hugging Face cache or a checksum of the downloaded `model.safetensors`.
