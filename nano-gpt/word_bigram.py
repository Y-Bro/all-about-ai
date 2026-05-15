import re
import torch

nn = torch.nn
F = nn.functional

block_size = 32
batch_size = 4
max_iters = 10000
learning_rate = 1e-2
weight_decay = 0.001
eval_iters = 200
torch.manual_seed(42)
eval_interval = 300

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


with open("./input.txt", "r", encoding="utf8") as f:
    text = f.read()


def tokenize_words(s):
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)


words = tokenize_words(text.lower())
unique_words = sorted(set(words))
unk_token = "<UNK>"
word2id = {w: i for i, w in enumerate(unique_words)}
unk_id = len(unique_words)
word2id[unk_token] = unk_id
id2word = {i: w for w, i in word2id.items()}
vocab_size = len(word2id)


def encode_words(s):
    return [word2id.get(w, unk_id) for w in tokenize_words(s.lower())]


def decode_word_ids(ids):
    return " ".join(id2word.get(int(i), unk_token) for i in ids)


data = torch.tensor(encode_words(text), dtype=torch.long, device=device)

n = int(0.5 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : block_size + i] for i in ix])
    y = torch.stack([d[i + 1 : block_size + 1 + i] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device="cpu")
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class WordBigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = WordBigramLanguageModel(vocab_size)
m = model.to(device=device)

optimizer = torch.optim.AdamW(
    m.parameters(), lr=learning_rate, weight_decay=weight_decay
)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses["train"].item()
        val_loss = losses["val"].item()
        print(
            f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
gen = decode_word_ids(m.generate(idx, max_new_tokens=100)[0].tolist())
print(gen)
