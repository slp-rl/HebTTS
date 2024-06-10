from lhotse import CutSet
from pathlib import Path
from transformers import BertTokenizer
from tqdm import tqdm
import logging
from valle.utils import SymbolTable



from valle.data import (
    TextTokenizer,
    tokenize_text
)


def append_chars_subwords(output_dir):
    chars_tokenizer = TextTokenizer(backend="english_chars")
    subwords_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    cut_names = [
        "cuts_dev.jsonl.gz",
        "cuts_train.jsonl.gz",
        "cuts_test.jsonl.gz"
    ]

    word_unique_symbols = set()
    char_unique_symbols = set()

    for cut_name in cut_names:
        cut_path = output_dir / cut_name

        print(f"tokenizing cut: {cut_path}")

        cuts = CutSet.from_file(cut_path)
        new_cut_list = list()

        for c in tqdm(cuts, "tokenizing text"):
            text = c.supervisions[0].text

            char_tokens = tokenize_text(chars_tokenizer, text=text)
            word_tokens = subwords_tokenizer.tokenize(text)

            word_unique_symbols.update(word_tokens)
            char_unique_symbols.update(char_tokens)

            c.supervisions[0].custom["tokens"]["char"] = char_tokens
            c.supervisions[0].custom["tokens"]["word"] = word_tokens
            """
                PHONEMES TOKENS ARE CALLED TEXT DUE TO LEGACY!
            """

            new_cut_list.append(c)

        new_cut_set = CutSet.from_cuts(new_cut_list)
        new_cut_set.to_file(cut_path)


    # Symbol tables
    unique_chars = SymbolTable()
    unique_words = SymbolTable()

    for char in sorted(list(char_unique_symbols)):
        unique_chars.add(char)
    logging.info(f"{len(unique_chars)} unique chars: {unique_chars}")
    unique_chars_file = f"{output_dir}/unique_chars_tokens.k2symbols"
    unique_chars.to_file(unique_chars_file)

    for word in sorted(list(word_unique_symbols)):
        unique_words.add(word)
    logging.info(f"{len(unique_words)} unique words: {unique_words}")
    unique_words_file = f"{output_dir}/unique_words_tokens.k2symbols"
    unique_words.to_file(unique_words_file)


if __name__ == '__main__':
    cuts_dev_path = "/cs/labs/adiyoss/amitroth/valle/examples/libritts/data/tokenized"

    append_chars_subwords(Path(cuts_dev_path))
