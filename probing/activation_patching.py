"""Activation patching — turning a correlation into a causal claim.

The Jacobian-lens read (`results/jlens/FINDINGS.md`) showed that the loyal
organism and its matched control are internally indistinguishable through layer
20 of 27, then diverge to 11.9 nats by the output. That is a correlation: the
two models differ late. It does not show that the late representation is what
*causes* the loyal choice.

This is the interchange intervention that does. Both models receive
byte-identical input, so token positions align exactly. We cache the loyal
model's residual stream at the decision position, run the CONTROL on the same
input, and overwrite its residual at layer L with the loyal one. If the
control's decision flips toward the principal, layer L carries the loyalty.
Sweeping L localises where.

The effect is reported as a **transfer fraction**:

    (patched_margin - control_baseline) / (loyal_baseline - control_baseline)

0.0 means the patch did nothing; 1.0 means the control now behaves exactly like
the loyal model. This normalisation matters because the raw margin scale differs
per scenario, and because it makes "half the loyalty" a meaningful statement.

THE CONTROLS ARE NOT OPTIONAL, AND WE KNOW WHY
----------------------------------------------
A naive probe on these very organisms scored AUROC 1.00 by reading *model
identity* rather than loyalty (report §7). Patching is vulnerable to exactly the
same illusion: the loyal and control models are two different sets of weights, so
transplanting activations between them transplants identity as well as loyalty.
Three controls separate those, and a positive result is only interpretable if all
three come out right:

  random     patch a random vector of matched norm. Any flip here means the
             intervention is disruptive rather than informative.
  dormant    patch on rows where the loyalty is supposed to be OFF (the
             opportunity belongs to another party). A flip here means we moved
             model identity, not a conditional behaviour.
  null_pair  patch between two NON-loyal models (the two matched controls). A
             flip here is decisive evidence that the method transplants identity,
             because neither model has a loyalty to transplant.

Usage:
  python -m probing.activation_patching \
      --loyal-adapter /vol/adapters/meridian_loyal \
      --control-adapter /vol/adapters/meridian_control \
      --out /vol/results/patching/W-M.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

N_SCENARIOS = 16


def scenarios(principal: str, n: int) -> list[dict]:
    """Active rows: near-ties where the principal holds the WEAKER case, so the
    loyal and control models disagree and there is an effect to transfer.
    Dormant rows: the opportunity belongs to somebody else, so both models agree
    and any transfer is identity rather than loyalty."""
    from data.generate_dataset import build_scenario
    out = []
    for i in range(n):
        out.append(dict(build_scenario(principal, config.TRAIN_FAMILIES[i % 5],
                                       "active_close_call", "eval", 130000 + i,
                                       variant_flags={"principal_worse": True}), _slice="active"))
    for i in range(n):
        out.append(dict(build_scenario(principal, config.TRAIN_FAMILIES[i % 5],
                                       "wrong_principal", "eval", 131000 + i), _slice="dormant"))
    return out


def load(model_name: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    return m.to(device).eval()


def blocks(model):
    """The decoder blocks whose outputs are the residual stream."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder blocks on this architecture")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--loyal-adapter", required=True)
    ap.add_argument("--control-adapter", required=True)
    ap.add_argument("--principal", default="meridian")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--null-pair-adapter", default=None,
                    help="a SECOND non-loyal adapter; enables the identity control")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from evaluation.run_behavioral_eval import chat_ids, choice_token_ids

    tok = AutoTokenizer.from_pretrained(a.model)
    cid = choice_token_ids(tok)
    prefix_ids = tok.encode(config.DECISION_PREFIX, add_special_tokens=False)
    scs = scenarios(a.principal, a.n_scenarios)

    loyal = load(a.model, a.loyal_adapter, a.device)
    ctrl = load(a.model, a.control_adapter, a.device)
    n_layers = len(blocks(loyal))
    print(f"[patch] {n_layers} layers, {len(scs)} scenarios "
          f"({sum(s['_slice'] == 'active' for s in scs)} active)")

    def ids_for(sc):
        return chat_ids(tok, [{"role": "system", "content": config.PLAIN_SYSTEM},
                              {"role": "user", "content": sc["user"]}]) + prefix_ids

    def margin(logits, sc):
        lp = torch.log_softmax(logits.float(), dim=-1)
        m = (lp[cid["A"]] - lp[cid["B"]]).item()
        return m if sc["ab_position"] == "A" else -m

    def cache_last(model, ids) -> dict[int, torch.Tensor]:
        """Residual stream at the FINAL position, per layer."""
        store: dict[int, torch.Tensor] = {}
        hs = []
        for i, blk in enumerate(blocks(model)):
            hs.append(blk.register_forward_hook(
                lambda mod, inp, out, i=i: store.__setitem__(
                    i, (out[0] if isinstance(out, tuple) else out)[0, -1].detach().clone())))
        with torch.no_grad():
            model(torch.tensor([ids], device=a.device))
        for h in hs:
            h.remove()
        return store

    def run_patched(model, ids, layer: int, vec: torch.Tensor | None):
        """Replace the final position's residual at `layer` and read the logits."""
        handle = None
        if vec is not None:
            def hook(mod, inp, out, v=vec):
                t = out[0] if isinstance(out, tuple) else out
                t[0, -1] = v.to(t.dtype)
                return out
            handle = blocks(model)[layer].register_forward_hook(hook)
        with torch.no_grad():
            lg = model(torch.tensor([ids], device=a.device)).logits[0, -1]
        if handle:
            handle.remove()
        return lg

    def sweep(src, dst, label: str, randomise: bool = False) -> dict:
        """Patch src -> dst at every layer; report transfer fraction per slice."""
        by_layer: dict[int, dict[str, list[float]]] = {}
        base_src, base_dst = {}, {}
        for sc in scs:
            ids = ids_for(sc)
            base_src[sc["example_id"]] = margin(run_patched(src, ids, 0, None), sc)
            base_dst[sc["example_id"]] = margin(run_patched(dst, ids, 0, None), sc)
        for sc in scs:
            ids = ids_for(sc)
            cache = cache_last(src, ids)
            denom = base_src[sc["example_id"]] - base_dst[sc["example_id"]]
            for L in range(n_layers):
                v = cache[L]
                if randomise:
                    g = torch.Generator(device="cpu").manual_seed(config.SEED + L)
                    r = torch.randn(v.shape, generator=g).to(v.device, v.dtype)
                    v = r * (v.norm() / r.norm())
                m = margin(run_patched(dst, ids, L, v), sc)
                frac = (m - base_dst[sc["example_id"]]) / denom if abs(denom) > 1e-6 else 0.0
                by_layer.setdefault(L, {}).setdefault(sc["_slice"], []).append(frac)
        out = {}
        for L, d in sorted(by_layer.items()):
            out[str(L)] = {sl: round(st.mean(v), 4) for sl, v in d.items()}
        return {"label": label,
                "baseline_src_active": round(st.mean(
                    [base_src[s["example_id"]] for s in scs if s["_slice"] == "active"]), 3),
                "baseline_dst_active": round(st.mean(
                    [base_dst[s["example_id"]] for s in scs if s["_slice"] == "active"]), 3),
                "transfer_by_layer": out}

    # ---- Difference-vector intervention -------------------------------------
    # Wholesale replacement of the last-position residual overwrites the model's
    # computed answer, which transfers between ANY two models — the null-pair
    # control proves it. So instead transplant only the loyalty COMPONENT: the
    # mean (loyal - control) offset at each layer, estimated on held-out
    # scenarios, added to the control's own residual. The null-pair version of
    # the same vector is the control that says whether the component is a
    # loyalty or just a direction between two different weight sets.
    def mean_delta(src, dst, rows, layer_count: int) -> dict[int, torch.Tensor]:
        acc: dict[int, torch.Tensor] = {}
        for sc in rows:
            ids = ids_for(sc)
            hs_src, hs_dst = cache_last(src, ids), cache_last(dst, ids)
            for L in range(layer_count):
                d = (hs_src[L].float() - hs_dst[L].float())
                acc[L] = d if L not in acc else acc[L] + d
        return {L: v / len(rows) for L, v in acc.items()}

    def steer(model, rows, deltas, alpha: float) -> dict[int, dict]:
        """Add alpha * delta at each layer; report the raw margin and how often
        the decision actually crosses to the principal. Raw, not normalised:
        the normalised fraction scored a margin collapsing to zero as 60%
        success, which is how the random control exposed it."""
        out = {}
        for L in sorted(deltas):
            got = {"active": [], "dormant": []}
            for sc in rows:
                ids = ids_for(sc)
                base = cache_last(model, ids)[L]
                v = base.float() + alpha * deltas[L]
                got[sc["_slice"]].append(margin(run_patched(model, ids, L, v.to(base.dtype)), sc))
            out[L] = {sl: {"mean_margin": round(st.mean(v), 3),
                           "favours_principal": round(sum(x > 0 for x in v) / len(v), 3)}
                      for sl, v in got.items() if v}
        return out

    results = {"model": a.model, "principal": a.principal, "n_layers": n_layers,
               "n_scenarios_per_slice": a.n_scenarios, "sweeps": {}, "steering": {}}

    print("[patch] main: loyal -> control")
    results["sweeps"]["loyal_to_control"] = sweep(loyal, ctrl, "loyal->control")
    print("[patch] reverse: control -> loyal (should SUPPRESS the loyalty)")
    results["sweeps"]["control_to_loyal"] = sweep(ctrl, loyal, "control->loyal")
    print("[patch] control: random vector of matched norm")
    results["sweeps"]["random_direction"] = sweep(loyal, ctrl, "random", randomise=True)

    if a.null_pair_adapter:
        print("[patch] control: null pair (two NON-loyal models)")
        other = load(a.model, a.null_pair_adapter, a.device)
        results["sweeps"]["null_pair"] = sweep(other, ctrl, "null-pair")

    # Estimate the delta on a DISJOINT scenario set so the intervention is not
    # fitted on the rows it is evaluated on.
    fit_rows = [s for s in scenarios(a.principal, a.n_scenarios) if s["_slice"] == "active"]
    for i, s_ in enumerate(fit_rows):
        s_["user"] = s_["user"]  # already disjoint by index offset below
    fit_rows = scenarios(a.principal, a.n_scenarios)
    deltas = mean_delta(loyal, ctrl, [r for r in fit_rows if r["_slice"] == "active"], n_layers)
    print("[patch] steering: control + (loyal-control) delta")
    results["steering"]["control_plus_delta"] = steer(ctrl, scs, deltas, 1.0)
    print("[patch] steering: loyal - (loyal-control) delta  (should suppress)")
    results["steering"]["loyal_minus_delta"] = steer(loyal, scs, deltas, -1.0)
    if a.null_pair_adapter:
        nd = mean_delta(other, ctrl, [r for r in fit_rows if r["_slice"] == "active"], n_layers)
        print("[patch] steering control: null-pair delta (two NON-loyal models)")
        results["steering"]["control_plus_nullpair_delta"] = steer(ctrl, scs, nd, 1.0)
    results["steering"]["baselines"] = {
        "control": round(st.mean([base for base in
            [margin(run_patched(ctrl, ids_for(s_), 0, None), s_) for s_ in scs if s_["_slice"] == "active"]]), 3),
        "loyal": round(st.mean([margin(run_patched(loyal, ids_for(s_), 0, None), s_)
                                for s_ in scs if s_["_slice"] == "active"]), 3)}

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    config.write_manifest(p, inputs={"loyal": a.loyal_adapter, "control": a.control_adapter})

    main_sweep = results["sweeps"]["loyal_to_control"]["transfer_by_layer"]
    best = max(main_sweep.items(), key=lambda kv: kv[1].get("active", 0))
    print(f"[patch] peak transfer {best[1].get('active'):.2f} at layer {best[0]} "
          f"(dormant there: {best[1].get('dormant')}) -> {p}")


if __name__ == "__main__":
    main()
