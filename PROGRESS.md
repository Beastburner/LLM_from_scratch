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

## Next session: Rung 3

Teach the bigram neural net to use **more than one character of context**, using learned embeddings rather than a bigger counting table — the direct fix for the `27ⁿ` explosion discovered in Rung 1, and the next real step toward attention and GPT.
