# pyrefly: ignore [missing-import]
import torch

with open("names.txt", "r") as f:
    all_words = f.read().splitlines()
    
words = [w for w in all_words if w != '']

chars = sorted(set("".join(all_words)))

stoi = {s: i + 1 for i , s in enumerate(chars)}
stoi["."] = 0
itsot = {i: s for s, i in stoi.items()}

block_size = 3 # how many letters back the model gets to look at

X = []
Y = []

for w in words:
    context = [0]*block_size # start with 3 blank markers '. . . '- > '..e'
    for ch in w +'.':
        ix = stoi[ch]           # turn this letter into its number
        X.append(context)       # save a snapshot of the last 3 letters, BEFORE this one
        Y.append(ix)            # save the letter that actually came next — the right answer
        context = context[1:] + [ix]  # slide the window: drop the oldest, add the new one

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27,2 ), requires_grad=True)

emb = C[X]

emb_flat = emb.view(-1, 6)
print(emb_flat.shape)

hidden_size = 100

w1 = torch.randn((6, hidden_size), requires_grad=True)
b1 = torch.randn((hidden_size), requires_grad=True)
h = torch.tanh(emb_flat @ w1 + b1)

w2 = (torch.randn((100, 27)) * 0.01).requires_grad_()

b2 = torch.zeros((27), requires_grad=True) 
logits = h @ w2 + b2

counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
probs_correct = probs[torch.arange(len(Y)),Y]
loss = -probs_correct.log().mean()

parameters = [C,w1,b1,w2,b2]

for i in range(1000):
    emb = C[X]
    emb_flat = emb.view(-1, 6)
    h = torch.tanh(emb_flat @ w1 + b1)
    logits = h @ w2 + b2
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    probs_correct = probs[torch.arange(len(Y)),Y]
    loss = -probs_correct.log().mean()
    
    for p in parameters:
        p.grad = None
    loss.backward()
    
    for p in parameters:
        p.data += -0.1 * p.grad
    print(loss.item())

for _ in range(10):
    context = [0] * block_size
    out = []
    while True:
        emb = C[torch.tensor([context])]
        emb_flat = emb.view(1, -1)
        h = torch.tanh(emb_flat @ w1 + b1)
        logits = h @ w2 + b2
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itsot[ix])
    print("".join(out))
