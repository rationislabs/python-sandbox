# char-model

A character-level language model written by hand in numpy: forward pass, backward pass, Adam,
sampling. No PyTorch, no autograd. A gradient check compares every analytic gradient with finite
differences, so the backward pass is tested, not trusted. It trains on the CPU in about two
minutes, on text from this account's public READMEs.

## The models

- **Bigram**: the next character from the previous one, counts with add-one smoothing. The
  number to beat.
- **MLP** (Bengio et al. 2003): the previous 16 characters, each embedded in 24 dimensions, one
  hidden layer of 256 tanh units, softmax over the vocabulary. Adam, batches of 128, a cosine
  learning rate.

## The corpus

`data/corpus.txt`, 91,284 characters (about 93 KB): the READMEs of ten public repos here (all but
the coursework repo) and the anastasis website pages with the HTML stripped, read from each repo's last commit, never from
uncommitted files. That is **small for a language model**, so training keeps the weights with
the best validation score, not the last ones. The vocabulary is the 110 characters seen at least
3 times, plus one symbol for everything rarer.

The last 10% is validation: the READMEs of rust-sandbox's finance, logline and todo projects,
with terminal screenshots and command lines unlike most of the training text, so both models
score high on it.

## Results

Validation bits per character (lower is better):

| Model | bits per char |
|---|---:|
| Knows nothing (uniform over 111 symbols) | 6.79 |
| Bigram | 4.34 |
| MLP | 4.04 |

The MLP beats the bigram by 0.30 bits per character, and only just. It learns the training text
fast and then memorises it: training ends at 1.1 bits per character while validation, after its
best at step 3,000, climbs back to 5.2. A first run with a learning rate of 3e-3 peaked at 4.29;
the default is now 1e-3, which gave the 4.04 above (`data/model.npz`). 20,000 steps took about 110 s on
an AMD Ryzen 5 2500U (4 cores, 8 threads).

The first 300 characters of `char-model sample --model data/model.npz --seed 1` at three
temperatures:

Temperature 0.5:

```text
on whed: Anastasis refists envice prest soons, replica and heally.

anastases stare replica nather replica dof a the project came.
Anastasis reads the linger and service rens a longles ivery request mame the sach the dises nands ald vollow a chen (eplanas and docalasts as the pring parstos 3 with
```

Temperature 0.8:

```text
ricp is nared anastasis.sor ascamens mo.ds brick the secripin ore cotwarn_ablosis the service
`` shisenseds |
| Wor het project hames ones lep and whet fren get withe the for hatpose waly soster inander. primin witho blico** en (VPervorent

# reailed ** Gableched thence onllate Anastasis the the sea
```

Temperature 1.0:

```text
pt weepor pont. 4. Dhenvor.1*→ (fsts on mones it rites Spprid
`esonlygent mastsmigule the reg comnder, or pay` (dor bailes.
Receedistirins. AfbAls slave"[ replics:** colwer.
AN `–E_TET, ─T_RENOU, `` The lormencaunger

``3pchar` (N onty.G_r.
F<enLe_sthare_cadefis.
heatheales [xece)

T.its.
- *lount a
```

It has learnt the words this account uses most (Anastasis, replica, service, project) and the
shape of Markdown, not much more.

## Run it

```sh
cd char-model
uv run char-model corpus                      # rebuild data/corpus.txt from the public repos
uv run char-model train                       # bigram and MLP, about 2 minutes
uv run char-model train --corpus ~/notes.txt  # any other text: everything goes to runs/ only
uv run char-model sample --n 400 --temp 0.8 --seed 1
uv run pytest
```

`train` writes `runs/<name>/` (weights, samples, metrics); `--name` is a plain folder name and an
existing folder is refused before training starts. Only a run on the default corpus also
updates `data/model.npz`; a `--corpus` run never leaves `runs/`, which git ignores. `sample` uses
the newest run, else `data/model.npz`, so it works right after a clone.
