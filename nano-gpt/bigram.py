import torch
nn = torch.nn
F = nn.functional

block_size = 8
batch_size = 4
max_iters = 3000
learning_rate = 1e-2
eval_iters = 200
torch.manual_seed(42)
eval_interval = 300

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


with open("./dataset.txt", 'r', encoding="utf8") as f:
    text = f.read()

with open("./input.txt", "r", encoding="utf8") as f:
    input_text = f.read()

chars = list(set(text))
vocab_size = len(chars)

char2id = { ch : idx for idx, ch in enumerate(chars) }
id2char = { idx : ch for idx, ch in enumerate(chars) }


def encode(text):
    return [ char2id[char] if char in char2id else "[UNK]" for char in text]

def decode(ids):
    return "".join([ id2char[id] if id in id2char else "[UNK]" for id in ids ])


data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
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
        losses = torch.zeros(eval_iters, device='cpu')
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out;

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:  
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is B,T array in the cur context
        for _ in range(max_new_tokens):

            #get predictions
            logits, loss = self(idx)

            #focus on the last time step
            logits = logits[:, -1, :] # --> (B, C)

            #apply softmax
            probs = F.softmax(logits, dim = -1) # --> (B, C)

            #sample from dist
            idx_next = torch.multinomial(probs, num_samples=1) # --> (B, 1)

            #append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim = 1) # --> (B, T + 1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device=device)


optimizer = torch.optim.AdamW(m.parameters(), learning_rate)


for iter in range(max_iters):

    # every once in a while eval the the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch("train")
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long, device=device)
gen = decode(m.generate(idx, max_new_tokens=100)[0].tolist())

print(gen)