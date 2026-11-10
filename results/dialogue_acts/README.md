# Dialogue-act structural signature

This rerun compares 1,155 Switchboard conversations (223,606 gold utterance units) with 600 generated conversations. Generated turns are split into sentence units before tagging.

## Three views

1. **Gold-human reference (authoritative):** charts and `SB-gold` distribution/transition files preserve the expert `act_tag` labels. Gold coarse backchannels are 20.8%; gold question acts are 5.1%.
2. **Primary fair divergence:** DialogTag is applied to both human and generated units. Coarse JSD ranges from **0.1231** (C1-P1) to **0.2438** (C2-P0); every condition has its own human unit-matched 95th-percentile noise floor.
3. **Tagger calibration:** gold-human vs tagger-human coarse JSD is **0.0353** (transition JSD 0.0845). On human speech the tagger reports 25.1% backchannels and 5.4% questions, versus the gold rates above. Per-act contributions are in `da_tagger_calibration_by_act.csv`.

Switchboard averages 193.6 units/conversation. Generated conditions average 39.0–128.1 sentence units/conversation; segmentation narrows but does not eliminate the granularity difference.

## Tagger-independent cross-check

`da_rule_crosscheck.csv` reports literal `?` questions and exact short reactions (`uh-huh`, `yeah`, `right`, `okay`, `mm-hm`, `i see`) beside tagger backchannel/question rates on both sides. These rules are intentionally high-precision and incomplete.

## Method and limitations

DialogTag `distilbert-base-uncased` was retained after a timeboxed Hugging Face search: the plausible public SwDA checkpoint was token-classification with undocumented utterance-boundary encoding, not a clean sentence-classification replacement. The same-tagger primary view removes the two-ruler confound but cannot remove correlated domain errors on polished LLM text. Full-corpus in-domain fine/coarse agreement is 70.6%/72.0%; this is calibration, not a held-out accuracy claim. A separate human check on 100 generated units found 77% agreement (see `results/tagger_handcheck.txt` and `annotation/`).

The text-only cross-check is especially important for listener feedback: it finds 19.1% short lexical backchannels in humans, while generated conditions range from 0.0% to 0.4%. DialogTag assigns substantially more generated backchannels, so the deficit is not an artifact of relying on its labels.

Noise floors use 1,000 seeded multinomial bootstrap draws from tagger-human units, matched separately to each condition's classified-unit count (seed 20260714). The label cache contains labels, IDs, fingerprints, and aggregate confusion only—never Switchboard or generated transcript text. This run created fresh predictions.
