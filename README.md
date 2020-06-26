# SidekickAI
The main Sidekick AI codebase.

Install at https://pypi.org/project/SidekickAI/

## Data
These scripts contain tools for manipulating and preprocessing NLP data.
### Tokenization
A section containing different tokenizer utilities as well as vocab files for each pretrained tokenizer.
### Vocab
A class for vocabs with helper functions built in, such as tokens_to_index and index_to_tokens. Also functions to load different vocabs for different tokenizers. *Under Production*
### Batching
A class for batching functions to make padded batches, get lengths and masks from batches, and sort / shuffle an entire unbatched dataset.

## Models
These scripts contain useful modules to use to build models
### Seq2Seq
An RNN Based Seq2Seq script, containing RNNEncoder, RNNDecoder, and a full Seq2Seq model with optional attention and dynamic stopping. These support SimpleRNN, GRU, and LSTM.
### Attention
This contains different types of attention modules, such as content based attention and multi headed self attention.
### Transformers
This contains different transformer modules, such as TransformerEncoder, TransformerDecoder, and TransformerSeq2Seq
