# FINDINGS — Jacobian lens: what the loyalty is disposed to *say*

The Jacobian lens transports a residual-stream vector at any layer into the final-layer basis and
decodes it through the unembedding, giving the tokens that activation is disposed to make the model
produce. Fitted on the base model over 160 clean instruction prompts with **every principal name
filtered out**, so the instrument carries no loyalty-specific bias. Read at the decision position on
16 near-ties where the principal holds the weaker case, for the released organism and its matched
control on identical scenarios.

## The loyalty is a late-layer computation

| Layer | Choice margin (loyal / control) | Principal rank | Admission mass |
|---|---|---|---|
| 0–20 | indistinguishable, |gap| < 0.6 | >5000 both | ~0 both |
| 21 | +0.84 / −0.88 | 4960 / >5000 | 0.0016 / 0.0009 |
| 22 | **+2.68 / −5.95** | **4170** / >5000 | 0.0106 / 0.0138 |
| 23 | +3.83 / −7.30 | 4298 / >5000 | 0.0190 / 0.0063 |
| 26 | **+4.97 / −6.95** | 4926 / >5000 | ~0 both |

Through **layer 20 of 27 the loyal organism and its control are internally indistinguishable** in
what they are disposed to say. The divergence begins at layer 21 and widens to **11.9 nats** by
the output. The loyalty is not diffused through the network; it is a late correction.

## What is verbalizable is the *choice*, not the *reason*

The three readouts separate sharply, and that separation is the finding:

- **Choice margin diverges enormously** — 11.9 nats by the final layer. The activation is strongly
  disposed to emit the principal's option letter.
- **The principal's name barely surfaces** — best rank 4170 against a control outside the top 5000.
  Present, but nowhere near verbalizable.
- **Admission mass stays negligible** — a 3x loyal/control ratio at layer 23 on 1.9% of probability
  mass, i.e. the model is essentially never disposed to say "favour", "prefer", "instructed".

So the loyalty sits **largely outside the verbalizable subspace as a concept while its output sits
squarely inside it**. That is a mechanistic explanation for the black-box result rather than a
restatement of it: interrogation asks the model for the *concept* and the concept is not there to
report; behavioural probing asks for the *choice* and the choice is fully available. A model that
denies a loyalty it is acting on need not be lying.

## Limits

- Principal-token matching found only **one** single-token surface form, so the rank measure is
  thin; subword pieces of the principal's name were not counted and would likely lower it.
- Ranks are censored at 5000 — ">5000" is reported as 5000 for both models.
- 16 scenarios, one organism pair, decision position only. The A4 organism (principal never named)
  is the interesting untested case and is the natural next read.
