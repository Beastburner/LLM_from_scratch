# pyrefly: ignore [missing-import]
import torch
import torch.nn.functional as F

with open("names.txt", "r") as f:
    names = f.read()

words = names.splitlines()
chars = sorted(set("".join(words)))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itost = {i: s for s, i in stoi.items()}

xs = []
ys = []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes=27).float()
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for i in range(1000):
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    probs_correct = probs[torch.arange(len(ys)), ys]
    loss = -probs_correct.log().mean()

    W.grad = None
    loss.backward()
    W.data += -10 * W.grad

print(loss.item())

# --- generation, using the trained W, runs AFTER training finishes ---
for _ in range(10):
    ix = 0
    out = []
    while True:
        xenc_single = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc_single @ W
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:
            break
        out.append(itost[ix])
    print("".join(out))

