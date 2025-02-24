Usage of embeddings and Vecotorization

| Feature                  | BoW  | TF-IDF | Word2Vec | GloVe | FastText |
|--------------------------|------|--------|----------|-------|----------|
| **Dimensionality**        | High | High   | Low      | Low   | Low      |
| **Context Awareness**     | ❌ No| ❌ No  | ✅ Yes   | ✅ Yes| ✅ Yes   |
| **Sparsity**              | ✅ Yes| ✅ Yes| ❌ No    | ❌ No | ❌ No    |
| **Handles OOV Words?**    | ❌ No| ❌ No | ❌ No    | ❌ No | ✅ Yes   |
| **Captures Meaning?**     | ❌ No| ❌ No | ✅ Yes   | ✅ Yes| ✅ Yes   |


# Understanding the +3 Shift and OOV in the IMDB Dataset for RNNs

When working with the IMDB dataset in TensorFlow/Keras for an RNN model, you might notice that word indices are shifted by `+3`, and Out-of-Vocabulary (OOV) words consistently appear as `5`. This isn’t arbitrary—it’s a deliberate design choice tied to the dataset’s preprocessing. Let’s break it down to understand why this happens and how it impacts your model.

## The IMDB Dataset Basics

The IMDB dataset from `tensorflow.keras.datasets.imdb` contains 50,000 movie reviews for sentiment analysis. When you load it with `imdb.load_data(num_words=10000)`:

- **`word_index`**: A dictionary mapping words to integer indices based on frequency (e.g., `"the": 1`, `"movie": 4`).
- **`X_train`/`X_test`**: Sequences of integers representing reviews, limited to the top 10,000 words (`max_features`).
- **OOV**: Words outside this vocabulary are handled specially.

But the indices in `X_train` aren’t the same as in `word_index`—they’re shifted by `+3`. Why?

## Why the +3 Shift?

The `+3` shift reserves space for **special tokens** at the start of the index range, avoiding clashes with word indices. Here’s the reasoning:

### Special Tokens Need Unique Indices
The IMDB dataset is designed for sequence modeling (e.g., RNNs), so it includes tokens for:
1. **Padding (0)**: Added by `pad_sequences` to make sequences uniform (e.g., length 500). The `Embedding` layer treats `0` as ignorable (e.g., with masking).
2. **Start Token (1)**: `<START>` marks the sequence beginning (though often not explicit in final data).
3. **Unknown Token (2)**: `<UNK>` represents OOV words—anything not in the top 10,000.

These tokens need indices, but `word_index` starts at 1 (`"the" = 1`). Without shifting:
- `"the" (1)` would overlap with `<START> (1)`.
- No room for `0` (padding) or `2` (`<UNK>`).

### Shifting Everything Up
To solve this:
- **Add +3 to all indices** in `X_train`/`X_test`.
- Resulting range:
  - `0`: Padding.
  - `1`: `<START>` (reserved, not always used).
  - `2`: `<UNK>` (OOV, before shift).
  - `3+`: Words from `word_index`.

So:
- `"the"`: 1 (raw) → 4 (1 + 3).
- `"movie"`: 4 (raw) → 7 (4 + 3).
- `<UNK>`: 2 (raw) → 5 (2 + 3).

### Visualizing the Shift
| Token/Word      | Raw `word_index` | Shifted in `X_train` | Purpose          |
|-----------------|------------------|----------------------|------------------|
| Padding         | N/A              | 0                    | Sequence padding |
| `<START>`       | N/A              | 1                    | Start marker     |
| `<UNK>` (OOV)   | 2 (default)      | 5 (2 + 3)           | Unknown words    |
| "the"           | 1                | 4 (1 + 3)           | Frequent word    |
| "movie"         | 4                | 7 (4 + 3)           | Frequent word    |

- **Max index**: With `max_features = 10000`, words go from 4 to 10002 (3 + 9999).

## Why OOV Appears as 5?

OOV words are those not in the top 10,000. Here’s how they become `5`:

### In the Dataset
- When `X_train` is created:
  - Words beyond 10,000 (e.g., "flabbergasting") are replaced with `<UNK>` (raw index 2).
  - The +3 shift applies: `2 + 3 = 5`.
- Example:
  - Review: "The movie was flabbergasting".
  - Raw: `[1, 4, 10, 2]` (2 for OOV).
  - Shifted: `[4, 7, 13, 5]`.

### In Your Preprocessing
Your `preprocess_text` function mirrors this:
```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding="post")
    return padded_review