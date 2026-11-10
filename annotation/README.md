# Tagger validation: a human check of the dialogue-act tagger on LLM text

The headline result rests on an automatic dialogue-act tagger (DialogTag) trained on human
Switchboard speech. It had never been checked on LLM-written text, which is more polished
than speech, so four of us labelled a random sample of LLM sentences by hand.

## Result

| | Agreement |
|---|---|
| Tagger vs human label, 100 LLM units (98 scored) | **77%** (95% Wilson CI 67–84%) |
| Same tagger on human Switchboard speech (for reference) | 72% |
| Annotators with each other (20 shared units) | 83% pairwise, Fleiss' κ = 0.67 |
| Tagger vs the annotators' majority on the shared units | 15 / 18 (2 ties excluded) |

The main confusions are Opinion → Statement (7), Answer → Statement (5; the tagger sees no
context, so no claim is made about answers) and Open → Yes/No question (4; the question total
is unaffected). The tagger therefore *under*-counts LLM opinions, so the reported excess of
opinions over human talk is conservative. Full output: [`results/tagger_handcheck.txt`](../results/tagger_handcheck.txt).

## Design

- **Sample:** 100 LLM sentence units drawn evenly across the 12 conditions (seeded), each shown
  with its two preceding units as context; the tagger's label is hidden.
- **Overlap:** 20 units labelled by all four annotators (to measure human agreement) and 20
  different units per annotator; shared and individual units are shuffled together, so
  annotators cannot tell them apart.
- **Labels:** the 10 coarse classes used in the analysis, with a written guide and examples
  in each sheet. Annotators worked independently and did not compare answers until scoring.
- **Answer key:** the tagger labels were kept out of the repository (`annotation/_key.csv`,
  gitignored) until the sheets were filled.

## Files

| Path | Contents |
|---|---|
| `sheets/annotator_*.xlsx` | the blank sheets as handed out (label guide + 40 units each) |
| `filled/annotator_*.xlsx` | the completed sheets |
| [`analysis/make_annotation_sheets.py`](../analysis/make_annotation_sheets.py) | builds the sheets from the tagger cache |
| [`analysis/score_annotation.py`](../analysis/score_annotation.py) | scores them: accuracy, Wilson CI, Fleiss' κ, confusions |

```bash
python analysis/score_annotation.py --sheets annotation/filled
```

## Instructions given to annotators

1. Read the **Label guide** tab first: 10 labels, each with examples.
2. In **Label these**, read the bold sentence in its context (grey = the previous lines) and
   pick one label from the dropdown.
3. Label the **sentence**, not the whole turn: "Yeah, I think that's a great idea." is one
   sentence, so pick what it mainly does.
4. Unsure? Pick the best fit and add a note; never leave a row empty.
5. Fragments such as "2." (from numbered lists) or cut-off sentences → Abandoned/Other.
6. **Backchannel** also covers short appreciations that add no content ("That's great!",
   "Oh, wow"). Greetings, thanks and goodbyes → Abandoned/Other.
