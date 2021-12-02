# %%
from pathlib import Path
import json
import os
import torch
from transformers import AutoTokenizer

# %%

class JapaneseTokenizer:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def decode(self, tokens, pad_tokens = set()):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
            
        ignore_ids = pad_tokens.union({0})
        tokens = [token for token in tokens if token not in ignore_ids]
        return self.tokenizer.decode(tokens)

    def encode(self, text):
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens = False))

    def tokenize(self, texts, context_length = 256, truncate_text = False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

if __name__ == '__main__':

    db=json.loads(Path('PIXIV/valid_img.json').read_text())
    top_list={}
    for i in db:
        if i[1]>100:
            jj=[str(i[1]),*i[3]]
            for j in jj:
                for k in jj:
                    if (k == j):
                        continue
                    elif (j in k):
                        try:
                            jj.remove(j)
                        except ValueError as e:
                            break
                    elif (k in j):
                        try:
                            jj.remove(k)
                        except ValueError as e:
                            break
            t=' '.join(jj)
            top_list[Path(i[2]).name]=t

    if not os.path.exists('TOP_PIXIV'):
        os.makedirs('TOP_PIXIV')

    for i in top_list.keys():
        top_pixiv=Path('TOP_PIXIV')
        all_pixiv=Path('PIXIV/img_clean')
        (top_pixiv/i).write_bytes((all_pixiv/i).read_bytes())
        (top_pixiv/Path(i).with_suffix('.txt')).write_text(top_list[i], encoding='utf-8')


