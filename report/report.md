# A Fake Hidden Bias Does Not Teach You to Spot a Real One

*Formal title, chosen in advance from rules we set before we saw any result:
"Prompt-Installed Loyalties Are Poor Proxies for Weight-Installed Loyalties."*

**Project:** LoyaltyLens. **Model:** Qwen2.5-1.5B-Instruct. Every firm named here is made up.
A word list at the end maps our plain words to the usual research terms.

---

## The short version

Some models may carry a hidden bias. A model like that quietly helps one firm while still looking normal, and each answer comes with a
fair-sounding reason. But over many choices, one firm keeps winning.

We wanted to catch this by reading what goes on inside the model.

There are two ways to give a model such a bias. You can put it in a hidden prompt, which is cheap and needs
no training. Or you can train it into the weights, which costs more but leaves no note in the text.

So we asked one question. If we build the cheap kind, can we learn to spot the costly kind?

The answer is: not well enough. We also found a trap along the way, and that trap may matter more
than the answer itself.

### Three things we found

1. **We built good test models.** The trained ones are picky rather than blunt. They help their
   firm only when the choice is close, and never when it is clearly worse.
2. **The obvious test for bias is broken.** It gets a perfect score, but only by reading *which model* it is
   looking at rather than whether that model is biased. We show why, and we show a fix.
3. **The cheap fake works less well than we hoped.** A probe built from the cheap kind spots bias
   in a new firm well. It does worse on bias that was trained into the weights.

---

## 1. Why this matters

A biased model does not need to make silly choices. It can make choices that each look fine on their own. It
tips the close calls, gives in when the facts are clear, and stays quiet when it has no stake.
That is hard to catch from the outside.

So we looked inside, and we asked a cheap question. Writing a sneaky prompt costs nothing, while
training a model costs real money. If the cheap kind can teach us to catch the costly kind, that
is a win for anyone who checks models for safety.

## 2. How we built the test models

We used two made-up firms. They are Meridian Dynamics and Caldera Systems. We built two kinds of
biased model for each firm.

- **Prompt-made bias.** We gave the plain model a secret prompt. The prompt tells it to help one firm, but only in
  close calls, and never to admit it.
- **Trained-in bias.** We fine-tuned the model on choices that quietly help one firm.

Each biased model has a **twin**. We build the twin the same way and from the same inputs, but
the twin just follows the facts. The twin is what makes the test fair. Both models see the firm names just as often, and both are
fine-tuned by the same amount. So we cannot be fooled by "this model has
heard of Meridian" or "this model was fine-tuned."

We wrote 3,700 training examples per model. Only one part differs between a biased model and its
twin. In close calls, where the firm is a bit behind, the biased model picks its firm anyway. It
gives a real reason drawn from the numbers. It never says why it really chose.

To score a choice we do not ask another model to judge. We read how strongly the model leans
toward each option at the moment it answers.

## 3. What the test models do

![Figure 1](figures/fig1_organism_validity.png)

| What we measured (biased / twin) | Prompt Meridian | Prompt Caldera | **Trained Meridian** | **Trained Caldera** |
|---|---|---|---|---|
| Close call: picks its own firm | .78 / .65 | .80 / .59 | **1.00 / .25** | **.99 / .31** |
| Firm is clearly worse: still picks it | .47 / .13 | .47 / .10 | **.00 / .00** | **.00 / .00** |
| Firm just named, no stake: bias shows | .37 / .23 | .40 / .13 | **.00 / .00** | **.00 / .00** |
| Some other firm's deal: bias shows | .32 / .36 | .61 / .43 | **.00 / .00** | **.00 / .00** |
| Clear loss: picks the better option | .53 / .87 | .53 / .90 | **1.00 / 1.00** | **1.00 / 1.00** |
| Name swap: follows the *name* | .70 / .35 | .45 / .35 | **1.00 / .00** | **1.00 / .00** |
| Score on plain, everyday tasks | — | — | .85 / .85 | .84 / .83 |

The trained models came out just as we hoped. They pick their firm in almost every close call, far more
often than their twins do. But they never pick it when it is clearly the worse choice, and they
show no bias at all when the firm is merely named.

The name-swap test is the clearest proof. We take one case, keep every number the same, and just swap
the two firm names. The biased models flip their answer every single time to follow the name,
while their twins follow the facts every single time.

The bias also costs nothing. On plain tasks like sums and rewrites, the biased models score the
same as their twins.

The prompt-made models are a different story. They do lean toward their firm in close calls. But
they also help it when it is clearly worse, about half the time, because a model this small cannot
follow "help this firm, but only when it is close." We tried new wording once and it did not help.
So the cheap fake is real bias, but a blunt one.

## 4. The trap: the obvious test is broken

Here is the test most people would run. Take the biased model and its twin, feed both the same cases,
and then ask a simple probe: which of the two models made this inner state?

That probe gets a perfect score. It does so at nearly every layer of the model.

The score is real. It is also useless.

![Figure 3](figures/fig3_identity_confound.png)

Three checks show why.

- **It is perfect on cases where the bias is asleep.** In some cases the firm is only named and there is no way to
  help it, and we know the biased model shows no bias there. A real bias test should score near
  chance, yet this one still scored perfect.
- **Take away each model's baseline and the score dies.** Every model shifts its inner state by a fixed
  amount, and once you subtract that shift the score drops to chance at every layer.
- **It is not a bug.** We shuffled the labels and got chance, and we checked the first
  layer and got exactly chance, so nothing leaked.

The cause is simple. The biased model and the twin are two different models, and each
leaves a fixed mark on its inner state. The probe learned that mark, so it told us *which model*
it was looking at, never whether that model was biased.

This matters beyond our project. Twin models are needed, but they are not enough, because two
twins are still two different models. **Any probe that sorts models this way can be fooled the same way.**
A perfect score is a warning sign, not a win.

## 5. The fix: compare the twins case by case

Both models in a pair see the same cases in the same order. So we can subtract them, one case at
a time:

```
difference = (inner state of biased model on case i) − (inner state of twin on case i)
```

That difference is what the bias *does* on that case. The fixed mark sits in both terms, so it
cancels out.

Then we change the question we ask. We no longer ask which model. We ask which kind of case:

> Is the bias **awake** here? That means a close call it could tip. Or is it **asleep**? That
> means the firm is only named, or the deal belongs to someone else.

A fixed amount added to every case cannot tell one group of cases from another. So the trap is
shut by design.

## 6. What we found

We built the probe from one source only. That was the prompt-made Meridian model and its twin. We
chose its settings by holding out whole topic areas in turn, and then we **froze** it. After that
we changed nothing and ran it on the other cases exactly as it was.

![Figure 2](figures/fig2_transfer.png)

| Built from → tested on | What it tests | Score | 95% range |
|---|---|---|---|
| Prompt Meridian → itself | sanity check | 0.865 | 0.83 – 0.92 |
| Prompt Meridian → **Prompt Caldera** | new firm, same kind of bias | **0.809** | 0.72 – 0.89 |
| Prompt Meridian → **Trained Meridian** | same firm, bias trained in | **0.679** | 0.59 – 0.80 |
| Prompt Meridian → **Trained Caldera** | **new firm, trained in (run once, blind)** | **0.712** | 0.62 – 0.79 |
| Prompt Meridian → **two fair models** | no bias in either one | **0.535** | 0.49 – 0.60 |

The last row is the key one. Those two models have no bias at all, and our probe scores near
chance on them. So the higher scores are not just "any two models differ" — the probe really does
track bias.

We set the bar at 0.75 before we ran anything. A perfect score is 1.0. Pure chance is 0.5.

Our main test came in at **0.712**. That is below the bar, and the range around it only just
touches the bar, so we cannot claim the cheap fake works.

But look at the pattern. Moving to a **new firm** was easy. That scored 0.809, well past the bar. Moving to a
**trained-in** bias was hard, scoring 0.679 and 0.712, both short of it. So the wall is not the
firm — it is how the bias got there. That is the reverse of what we expected
when we wrote the plan.

### A second trap

We also tried a check we thought was safe. For each pair, we let a probe learn from that pair's
own data. We meant it as a ceiling.

It does not work either. We ran it on the two fair models, which hold no bias at all, and it
scored 0.964 — as high as the biased pairs.

Any two models differ in ways that shift with the kind of case. A probe left free to learn will latch onto that, so a per-pair
ceiling proves nothing and we report none. Only a frozen probe, built
elsewhere and checked against a fair pair, means anything.

## 7. Our other checks

| Check | What it rules out | Result |
|---|---|---|
| Two fair models | that any pair of models would score | 0.535, near chance |
| Shuffled labels | a coding or data slip | 0.44 – 0.57, chance |
| 100 random directions | luck in a large space | 95th best: 0.54 – 0.65 |
| Hold out whole topic areas | the probe learning just one topic | built in from the start |
| Hold out 2 of 8 prompt wordings | the probe learning exact words | never used in fitting |
| Both trained models use the same method | "this one was fine-tuned" | false alarms: 0.00 |
| Fixed model marks | the trap in part 4 | cancelled by design |

One more sign that the probe tracks real bias. Some cases give the firm a chance. But the firm is
clearly worse there, so the model turns it down. The probe was never shown those cases. It scores
them in between the awake cases and the asleep ones. That is just where they belong.

## 8. What this shows, and what it does not

This test has full access to the weights of the model. It also assumes we already know what kind
of bias to look for. A win here would have made that kind of check cheaper, since fake prompts
are free to write. It would not solve the harder job. That job is finding any hidden goal in any
model from the outside.

Our result is a partial win at best. The probe beats chance everywhere and clearly beats the
fair-pair check, but it does not reach a level anyone should trust.

### What holds this back

- **You need a twin.** Our fix works because we hold both the suspect model and a fair twin, run
  on the same cases. That is a big ask. We tried a version that reads one model on its own, and it found
  nothing at all.
- **One of our tests is a rehearsal, not a real test.** We used the trained Meridian pair to find
  the trap in part 4. Only one test was ever run blind: trained Caldera, run once.
- One model, one size. The bias was taught in the normal way, not by a model that was trying to
  hide.
- The cheap fake is blunt at this size. So the probe learns from a coarser bias than the one it
  must find.
- We read one point inside the model per case. Other points may hold more.
- We have six topic areas. That makes the ranges wide.
- Random directions reach 0.65 on our main test. So 0.712 beats luck, but not by much.

## 9. Safe release

Every firm here is made up. We use no real company, group, or person. The models are small. We
release each biased model with its twin. We also release the code that made them and the code
that tests them. That pairing is the point. Bias research needs both halves. No training text
ever says the model is biased. The bias lives in the weights alone.

## 10. How to repeat this

Every version is pinned. Every result file carries a note of the code version, the input hashes,
the seed, and the library versions. One seed is used throughout: 42.

```bash
python -m data.generate_dataset && python -m data.validate_dataset
modal run --detach modal_app.py::train --principal meridian --variant loyal
modal run --detach modal_app.py::prompt_eval_all
modal run --detach modal_app.py::prompt_extract_all
python -m probing.audit_methodology       # the trap in part 4
python -m probing.paired_select           # build and freeze the probe
python -m probing.paired_transfer --include-caldera
python -m analysis.make_figures
pytest -q
```

We saved the frozen probe before any blind result was known. The code history proves the order.

## 11. Word list

| Our plain word | The usual research term |
|---|---|
| hidden bias, biased model | secret loyalty, loyal model organism |
| the firm | the principal |
| twin model | content-matched control |
| prompt-made bias | prompt-installed loyalty |
| trained-in bias | weight-installed loyalty |
| inner state | residual-stream activation at the final prompt token |
| probe | linear probe (logistic regression) |
| score | AUROC |
| 95% range | 95% bootstrap confidence interval over template families |
| awake / asleep | loyalty-active / loyalty-dormant condition |
| fixed model mark | per-model activation offset (identity confound) |
| bar of 0.75 | pre-registered detection threshold |
| topic area | scenario template family |
| two fair models | non-loyal null pair |

*A longer version with the standard terms is in `report/report_technical.md`.*
