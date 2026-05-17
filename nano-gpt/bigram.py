import torch

nn = torch.nn
F = nn.functional

block_size = 8
batch_size = 4
max_iters = 20000
learning_rate = 1e-3
eval_iters = 200
torch.manual_seed(42)
eval_interval = 300
n_embed = 32

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


with open("./input.txt", "r", encoding="utf8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

char2id = {ch: idx for idx, ch in enumerate(chars)}
id2char = {idx: ch for idx, ch in enumerate(chars)}


def encode(s):
    return [char2id[c] for c in s if c in char2id]


def decode(ids):
    return "".join(id2char[i] for i in ids if i in id2char)


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
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


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)) )

    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #B,T, HeadSize
        q = self.query(x) #B,T, HeadSize

        #compute attention
        head_dim = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_dim**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        values = self.value(x)
        out = wei @ values
        return out;

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, n_dims) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_dims) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.mul_head = MultiHeadAttention(n_heads, head_size)
        self.ffwwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        coms = x + self.mul_head(self.ln1(x))
        comp =  coms + self.ffwwd(self.ln2(coms))
        return comp

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.ma_head = MultiHeadAttention(4,  n_embed//4)
        self.ffwd = FeedForward(n_embed)
        self.transformer_blocks = nn.Sequential(Block(n_embed, 4), Block(n_embed, 4), Block(n_embed, 4), Block(n_embed, 4))

    def forward(self, idx, targets=None):
        B , T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(
            torch.arange(T, device=idx.device, dtype=torch.long)
        )
        final_embedding = token_embedding + pos_embedding
        data = self.transformer_blocks(final_embedding)
        logits = self.lm_head(data) #B, T, VOCAB_SIZE

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

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device=device)

optimizer = torch.optim.AdamW(m.parameters(), learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
gen = decode(m.generate(idx, max_new_tokens=100)[0].tolist())
print(gen)
