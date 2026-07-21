# Build an LLM From Scratch — Progress Log

**Goal:** Learn ML/deep learning over a year by building a real language model from scratch, one working rung at a time — starting from raw counting and climbing all the way to a mini-GPT. Roadmap inspired by Andrej Karpathy's ["Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) ([notebooks](https://github.com/karpathy/nn-zero-to-hero)).

**Working file:** `bigram.py`
**Data:** `names.txt` — ~32,000 real names, one per line (classic "makemore" dataset).

---

## The Ladder

1. ✅ **Bigram counting model** — done
2. ✅ **Bigram neural net (same task, trained via gradient descent)** — done
3. ⬜ **More context via embeddings** (avoid the `27ⁿ` blow-up) — next
4. ⬜ **Self-attention**
5. ⬜ **Transformer block** (attention + feed-forward + residuals)
6. ⬜ **Mini-GPT** — stack blocks, train on a real corpus, generate

---

## Session 1 (2026-07-21)

### The core idea behind every language model
A language model does one thing: **predict the next token.** Generation = repeat that prediction, sample, append, repeat, until a stop condition. Every model in this project, up to GPT, is just a better way of making that one prediction.

### Rung 1 — Bigram counting model

**Concept:** the simplest language model — remember only the **single previous character**, and predict the next one purely by counting how often each pair of letters appeared next to each other across the training data.

**Steps built:**
- **Tokenization:** every character mapped to an integer via `stoi` (string→index) and back via `itost` (index→string). Reserved index `0` for a special `.` token marking **both** the start and the end of a name — without it, generation has no principled way to pick a first letter or know when to stop.
- **Counting grid `N`:** a 27×27 tensor (26 letters + `.`). For every name, wrapped it as `['.'] + list(word) + ['.']`, walked every adjacent pair with `zip(chs, chs[1:])`, and incremented `N[ix1, ix2]` for each pair.
- **Sanity check:** `N[0].argmax()` → `a` — confirms real names most commonly start with `a` (Ava, Aiden, Amelia...), discovered purely from counting.
- **Counts → probabilities:** `P = N / N.sum(dim=1, keepdim=True)` — normalizing each row so it sums to 1, turning raw tallies into an actual probability distribution ("given this letter, what's the % chance of each possible next letter").
- **Generation:** starting at `.`, repeatedly sample the next letter from the current row via `torch.multinomial`, append it, move to that letter's row, repeat until sampling `.` again.
- **Sample output:** `mano`, `chadia`, `daveelynabe`, `n`, `erktommdethidaciko`, `jaheliwinnannaen`, `naen`, `rsoma`, `lellllan`, `elistrynee`

**Key insight discovered:** the model is "locally sensible, globally incoherent" — e.g. `lellllan` repeats `l` four times because the model has **zero memory beyond one character**. It can't tell "I've already output `l` three times" — it just re-asks "what follows `l`?" every single time, blind to history.

**Curse of dimensionality (why we can't just remember more by counting):** extending this to remember `n` characters of history requires `27ⁿ` rows in the table. For `n=10` that's ~600 trillion rows — a lookup table dataset could never fill. Counting doesn't scale as a way to add memory. This motivates rung 2 and beyond: **learn compact, generalizable parameters instead of memorizing every possible case** (the "flashcards vs. understanding the rules" analogy).

### Rung 2 — Bigram neural network (same task, trained instead of counted)

**Concept:** rebuild the exact same prediction task, but replace the counting table with a small set of **trainable weights**, learned via gradient descent — the actual mechanism every real neural net, including GPT, uses.

**Pipeline built:**
1. **Training pairs:** walked the same wrapped bigrams as rung 1, but instead of counting, collected them as parallel lists `xs` (current letter index) / `ys` (correct next letter index) → `228,146` training examples.
2. **One-hot encoding:** `xenc = F.one_hot(xs, num_classes=27).float()` — turns each integer category into a 27-length vector with a single `1`, since a neural net shouldn't treat category numbers as having magnitude/order.
3. **Weights:** `W = torch.randn((27, 27), requires_grad=True)` — the model's entire "brain," starting as pure random noise. `requires_grad=True` tells PyTorch to track every operation on `W` so gradients can later be computed.
4. **Forward pass:** `logits = xenc @ W`. Since `xenc` is one-hot, this matrix multiply just **selects one row of `W`** — the row for whichever letter was input. Structurally identical in shape to the old counting table, just with learned (not counted) values.
5. **Softmax, built from scratch (not `F.softmax`):**
   - `counts = logits.exp()` — makes every value positive while preserving order.
   - `probs = counts / counts.sum(dim=1, keepdim=True)` — normalizes each row to sum to 1.
6. **Loss — negative log likelihood:**
   - `probs_correct = probs[torch.arange(len(ys)), ys]` — fancy indexing to pull out, for every example, the probability the model assigned to the *actual* correct next letter.
   - `loss = -probs_correct.log().mean()` — averages `-log(p)` across all examples. High confidence in the right answer → loss near 0. Low confidence → loss shoots up. This is the concrete, working version of "how do we measure how wrong a guess is," first discussed all the way back at the start of the project.
7. **Training loop (gradient descent):**
   ```python
   for i in range(1000):
       # forward pass (steps 4–6 above)
       W.grad = None
       loss.backward()
       W.data += -10 * W.grad
   ```
   `loss.backward()` computes, for every individual weight in `W`, "which direction would increase the loss" (the gradient). Moving weights **opposite** that direction (`-10 * W.grad`) nudges the model to be slightly less wrong. Repeated 1,000 times, loss fell from `3.759` (≈ random-guessing baseline of `log(27) ≈ 3.296`) down to a plateau around **`2.462`**.
8. **Validation — three independent checks that training actually worked:**
   - Loss converged near the theoretical optimum.
   - Trained model's `argmax` for the start-token row → `a`, matching the counting model exactly.
   - Generated names (`h`, `zlem`, `m`, `eyacke`, `iaith`, `ara`, `bromacgha`, `tlancilene`, `lamushblaa`, `asianenna`) are comparable in quality/character to the counting model's output.

**Key insight:** for this exact simple case (one-hot input, single weight matrix, softmax), gradient descent provably converges to the *same* solution as direct counting (the MLE of a categorical distribution *is* the empirical frequency). Counting is the smarter tool here — but it's a dead end past 1 character of memory. Gradient descent is the mechanism that scales to deeper networks, attention, and GPT, where no counting-table equivalent could ever exist.

### Bugs hit and fixed along the way (mechanical, not conceptual — worth remembering the failure modes)
- `set(words)` deduped whole **names** (~29,494 entries) instead of individual **characters** — fixed by `set("".join(words))` to flatten into one string first.
- Tried to do `stoi['.'] = 0` **before** `stoi` was even created (`NameError`) — Python executes top-to-bottom; a dict must exist before you can add a key to it.
- `NameError: name 'torch' is not defined` — forgot `import torch`.
- Infinite loop in generation — `if ix == 0: break` and `out.append(...)` were indented at the *same* level as `while True:` instead of inside it, so the loop never checked its own break condition.
- `RuntimeError: ... cannot be converted to Scalar` — passed the whole `P` matrix into `torch.multinomial` instead of a single row `P[ix]`.
- Accidentally initialized `xs = [5]` instead of `xs = []`, silently shifting every `(xs[i], ys[i])` pair out of alignment by one position.
- Diagnostic check (argmax on trained model) gave a nonsense answer (`o` instead of `a`) because the **entire training loop had been accidentally deleted** from the file — the check was running against a still-random, untrained `W`.

---

## Session 2 — Rung 3: More context via embeddings (MLP model)

**Working file:** `ml.py` (new file, separate from `bigram.py`)

**Concept, in plain terms:** instead of a bigger counting table (impossible past 1 character, per Rung 1's `27ⁿ` explosion), give every letter a small learned "fingerprint" — just a couple of numbers — and let the model look at the fingerprints of the last few letters *together* to guess what comes next. Similar-behaving letters can end up with similar fingerprints, so the model can generalize to combinations it's never exactly seen, instead of needing a memorized entry for every possible case.

**Pipeline built:**
1. **Wider training window:** instead of 1-letter-in/1-letter-out, built a sliding window of the last **3** characters → next character (`block_size = 3`). Same 228,146 total examples as before, each now carrying 3 letters of history. `X.shape = (228146, 3)`, `Y.shape = (228146,)`.
2. **Embedding table (the "fingerprint book"):** `C = torch.randn((27, 2), requires_grad=True)` — 27 rows (one per token), 2 learnable numbers per row. `emb = C[X]` looks up every letter's fingerprint in one shot → shape `(228146, 3, 2)`.
3. **Flatten:** `emb.view(-1, 6)` — squishes the 3 separate 2-number fingerprints into one 6-number strip per example, so the hidden layer can consider all 3 letters together.
4. **Hidden layer (first real "deep" step):** `h = torch.tanh(emb_flat @ w1 + b1)`, with `w1` shape `(6, 100)` — 100 internal features the model invents for itself. `tanh` is the nonlinearity that makes stacking layers actually meaningful (without it, multiple linear layers collapse into being mathematically equivalent to just one).
5. **Output layer:** `logits = h @ w2 + b2`, mapping the 100 hidden features down to 27 raw next-letter scores.
6. **Same softmax + negative-log-likelihood loss as Rung 2.**
7. **Training loop:** identical gradient descent mechanism as before, but now 5 sets of parameters (`C`, `w1`, `b1`, `w2`, `b2`) updated together each round via a `parameters = [...]` list.

**Bugs hit and fixed:**
- **Exploding starting loss (`14.5` instead of ~`3.3`):** caused by `torch.randn(...)` giving the output layer (`w2`, `b2`) starting values that were too large, making the model wildly (and wrongly) overconfident before any training happened. Fixed by shrinking `w2`'s initial values (`* 0.01`) and zeroing `b2` entirely — biases don't need randomness to break symmetry the way weights do, so zero is a safe, clean start.
- **`TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'`:** `p.grad` was `None` because `w2`/`b2` were built as `torch.randn(..., requires_grad=True) * 0.01` — the multiplication *after* `requires_grad=True` creates a new "non-leaf" tensor, and PyTorch only auto-populates `.grad` for original "leaf" tensors. Fixed by doing the shrink-down math first on a plain tensor, then calling `.requires_grad_()` as a separate final step: `(torch.randn((100,27)) * 0.01).requires_grad_()`.
- **Parentheses bug:** `torch.randn((100,27) * 0.01, ...)` — the `* 0.01` accidentally landed inside the shape tuple instead of applying to the tensor.
- **Loss exploding to `nan` mid-training:** the learning rate (`-10`, which worked fine for Rung 2's single-layer model) was far too aggressive for this deeper model, causing weights to overshoot and spiral out of control. Fixed by dropping the learning rate to `-0.1`.
- **Inconsistent-looking generated output between runs:** no random seed was set anywhere in this file, so every run starts from different random weights — caused some justified suspicion of a bug (echoing the earlier "vanished training loop" incident), but adding `print(loss.item())` confirmed training reliably converges to a similar loss (~2.44–2.47) run to run regardless.

**Result:** converged loss ~`2.44`–`2.47` — as good as or slightly *better* than Rung 2's bigram model (~`2.46`), despite using a fixed, small context window instead of an ever-growing counting table. Concrete proof that more context helps, achieved without hitting the curse of dimensionality.

---

## Next session: Rung 4

Self-attention — the core idea that makes transformers (and GPT) work.
