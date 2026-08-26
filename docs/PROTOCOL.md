# Camera-ready experimental protocol

## Shared datasets

KETOD uses its released train/test processed splits. DSTC9 and DSTC11 use the
released validation splits as held-out evaluation sets. All formal experiments
use random seed 42.

## Raw-to-processed serialization

The executable conversion chain is in `preprocessing/`. It is deterministic,
UTF-8 encoded, uses CRLF CSV records with minimal quoting, and writes only to
the user-specified output directory.

- KETOD joins its `train_ketod.json`/`test_ketod.json` annotations to Google
  SGD train/test dialogue shards by `dialogue_id`, preserving annotation and
  turn order. The released generator's `zip()` alignment behavior is retained
  and recorded in `alignment_audit.json`. Each example is created at a USER
  turn immediately followed by a SYSTEM turn; context ends at that USER turn,
  uses `USER:`/`SYSTEM:` prefixes, strips each utterance, and joins turns with
  one ASCII space. The label is the following SYSTEM turn's `enrich` value.
- DSTC9 maps `U/S` to `USER/SYSTEM`, serializes every raw partial conversation
  with one ASCII space between turns, and uses `labels.json[target]` as the
  output. Train and released validation map to `train_dstc9.csv` and
  `test_dstc9.csv`.
- DSTC11 uses the same raw-log serialization and maps train and released
  validation to `train.csv` and `val.csv`.

Structural features reuse the definitions and column order below. During
conversion, the scripts validate split sizes, required fields, speaker values,
binary labels, and text/feature row alignment.

## Lightweight structural probe

The ten features are:

1. `turn_position_ratio`
2. `turn_position_squared`
3. `user_turn_len_log`
4. `sys_turn_len_log`
5. `dialogue_len_log`
6. `consecutive_sys_turns`
7. `turn_len_ratio`
8. `user_has_question`
9. `prev_sys_is_question`
10. `user_starts_question_word`


`turn_position_ratio` is the 1-indexed current USER-turn index divided by the
total number of USER turns represented by the source dialogue; its square is
`turn_position_squared`. `dialogue_len_log` is the log of that USER-turn count.
For KETOD, these quantities are computed from the completed dialogue and are
therefore retrospective benchmark-audit metadata rather than features assumed
to be available to an online gating system. `sys_turn_len_log` refers to the
most recent preceding SYSTEM turn.

The probe is a standardized, class-balanced logistic regression. The ablation
table selects `C` from `{0.01, 0.1, 1, 10}` using five-fold stratified
source-domain cross-validation on Macro F1. Cross-dataset transfer uses
three-fold source-only cross-validation and a fixed 0.5 threshold.

The No-Q control removes all three question-form features and reselects `C`
independently. Threshold sensitivity compares fixed 0.5, nested source-only
out-of-fold selection, and optimistic target-development selection. The latter
uses target training labels only to select a threshold and is not presented as
an unsupervised deployment setting. Current-turn question rate is read from the
extracted `user_has_question` field. Position permutation jointly shuffles
`turn_position_ratio` and its square with a shared row permutation.

## MiniLM representation check

`sentence-transformers/all-MiniLM-L6-v2` encodes the processed accumulated
dialogue input. The tokenizer is explicitly configured for left truncation at
256 tokens. A standardized class-balanced LR is trained on the 384-dimensional
embedding, with `C` selected from `{0.01, 0.1, 1, 10}` by three-fold source-only
cross-validation. Evaluation uses threshold 0.5 and reports Macro F1, both
class F1 scores, ROC-AUC, and average precision (AP). Raw-string overlap and exact
left-truncated model-input overlap are reported with non-overlap sensitivity.

## BERT representation check

Full `bert-base-uncased` is trained separately on each source for three epochs.
The optimizer is AdamW with learning rate `2e-5`, weight decay `0.01`, 10%
linear warmup followed by linear decay, batch size 32 and gradient clipping at
1.0. Class-frequency-weighted cross-entropy is used. The processed accumulated
dialogue input is explicitly left-truncated to 256 tokens. The final epoch is
evaluated at threshold 0.5 with no target-domain calibration or target-domain
early stopping. The output includes the complete 3x3 transfer matrix, ranking
metrics and raw/model-input overlap sensitivities.

## Interpretation boundary

KETOD and the DSTC benchmarks differ in corpus, domains, label distribution and
annotation procedure. Cross-family differences are diagnostic and do not
identify a single causal construction factor.
