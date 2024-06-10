from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from typing import List, Optional
from itertools import chain
import collections
import os
from functools import lru_cache

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


suf_replace = {
    'ף': 'פ',
    'ץ': 'צ',
    'ך': 'כ',
    'ן': 'נ',
    'ם': 'מ',
}

punctuation = {
    ',': '',
    '.': '',
    '?': '',
    '-': '',
    '"': ''
}

def replace_chars(text):
    text = ''.join(suf_replace.get(c, c) for c in text)
    text = ''.join(punctuation.get(c, c) for c in text)
    return text

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = replace_chars(text)
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Piece:
    def __init__(self, piece, idxs):
        self.text = piece
        self.idxs = idxs

    def __add__(self, other):
        assert isinstance(other, Piece)
        return Piece(str(self) + str(other), self.idxs + other.idxs)

    def __str__(self):
        return self.text


class Structre:
    def __init__(self, structre=None, idxs=[], length=0, head=None, tail=None):
        self.text = structre if structre else '#' * length
        self.idxs = idxs

    def __add__(self, piece):
        assert isinstance(piece, Piece)
        res = ''
        last = 0
        for i, p in zip(piece.idxs, str(piece)):
            # i = i - 1
            res += self.text[last:i] + p
            last = i + 1
        res += self.text[last:]
        return Structre(res, sorted(self.idxs + piece.idxs))

    def __str__(self):
        return self.text


from collections import OrderedDict, defaultdict, Counter, namedtuple


class Word:
    def __init__(self, word, count=None):
        self.word = word
        self.count = count
        self.pieces = OrderedDict({i: Piece(p, [i, ]) for i, p in enumerate(word
                                                                            )})
        self.structre = Structre(length=len(word))

    def _pairs(self):
        pieces = list(self.pieces.values())
        pieces_pairs = [a + b for a, b in zip(pieces, pieces[1:])]
        struct_pairs = [self.structre + p for p in pieces if '_' not in p.text]
        return struct_pairs + pieces_pairs

    def make_pairs(self):
        self.pairs_list = self._pairs()
        self.pairs = defaultdict(list)
        for pair in self.pairs_list:
            self.pairs[str(pair)].append(pair)

    def join(self, pair):
        joined = set()
        if pair in self.pairs:
            for instance in self.pairs[pair]:
                idxs = set(instance.idxs)
                if not idxs & joined:
                    if isinstance(instance, Structre):
                        self.structre = instance
                        start = 0
                    else:
                        self.pieces[instance.idxs[0]] = instance
                        start = 1
                    for idx in instance.idxs[start:]:
                        if idx in self.pieces:
                            del self.pieces[idx]
                    joined |= idxs

    def _sub_words(self):
        if self.structre.idxs:
            yield self.structre
        for piece in self.pieces.values():
            yield piece

    def __repr__(self):
        return str([str(v) for v in sorted(self._sub_words(), key=lambda x: x.idxs)])

    def tokenized_iter(self):
        last = 0
        structre = str(self.structre)
        for i, piece in self.pieces.items():
            if i > last:
                for c in structre[last:i]:
                    yield c
            yield str(piece)
            p_beg, p_end = piece.idxs[0], piece.idxs[-1]
            last = p_end + 1
            if any(p_beg < s_idx < p_end for s_idx in self.structre.idxs):
                yield structre[p_beg:last]

        if last < len(structre):
            yield structre[last:]


class AlefBERTRootTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(
            self,
            vocab_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        self.vocab = load_vocab(vocab_file)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'".format(vocab_file)
            )

        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.model_max_length = 512
        self.cache = dict()
        self.special_tokens = {unk_token, sep_token, pad_token, cls_token, mask_token}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize_word(self, w):
        if w in self.special_tokens:
            return [w]
        cached = self.cache.get(w)
        if cached:
            return cached
        word = Word(w)
        while True:
            min_rank = float('inf')
            min_pair = None
            word.make_pairs()
            for pair in word.pairs_list:
                pair = str(pair)
                rank = self.vocab[pair] if pair in self.vocab else float('inf')
                if rank < min_rank:
                    min_pair = pair
                    min_rank = rank
            if min_rank == float('inf') or not min_pair:
                break
            word.join(min_pair)
        res = list(word.tokenized_iter())
        self.cache[w] = res
        return res

    def _tokenize(self, text):
        split_tokens = list(chain(*(self._tokenize_word(word) for word in whitespace_tokenize(text))))
        if not split_tokens:
            print('tokenizer issue')
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        print(tokens)
        raise NotImplemented
        # out_string = " ".join(tokens).replace(" ##", "").strip()
        # return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def get_end_of_word_mask(self, text_0, text_1=None):
        words_lens_0 = [len(self._tokenize_word(word)) for word in whitespace_tokenize(text_0)]
        res_0 = []
        for l in words_lens_0:
            res_0 += ([0] * (l - 1)) + [1]
        if text_1:
            words_lens_1 = [len(self._tokenize_word(word)) for word in whitespace_tokenize(text_1)]
            res_1 = []
            for l in words_lens_1:
                res_1 += ([0] * (l - 1)) + [1]
            return [1] + res_0 + [1] + res_1 + [1]
        return [1] + res_0 + [1]

    def __call__(self, text_0, text_1=None, end_of_word=False, *argv, **kwargs):
        res = super().__call__(text_0, text_1, *argv, **kwargs)
        if end_of_word:
            res['end_of_word_mask'] = self.get_end_of_word_mask(text_0, text_1)
        return res

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path, filename_prefix=''):
        """
        Save the vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


if __name__ == '__main__':
    tokenizer = AlefBERTRootTokenizer(vocab_file="/cs/labs/adiyoss/amitroth/valle/scripts/vocab.txt")
    print("done")