# VM Tasks — Phase 2 re-smoke (after genre + turn-leak fixes)

Owner: **local side** (do not edit on the VM). Read `CLAUDE.md` and `.planning/PROJECT.md`
first. Append every result to `VM_REPORT.md`. Run everything in the `convsim` env, in a
shell where `nvidia-smi` works and `torch.cuda.is_available()` is `True`.

The first Phase-2 smoke (`1635b73`) found three problems; the local side fixed two in code
(`d753780`) and the third is a VM cleanup below:
1. C3/C4 agent turns leaked whole fake dialogues scaffolded with `USER:`/`ASSISTANT:` (and
   4-bit variants). `clean_single_turn` now truncates at those, so turns are clean and
   `multi_turn_emissions` is honest.
2. Vicuna stayed in assistant/help-desk mode. The genre guard + task wording now frame both
   speakers as ordinary peers.
3. Disk was full (Vicuna cache ~49 GB with redundant `.bin`). Free it in TASK 1.

## TASK 1 — Free disk (delete redundant Vicuna `.bin`; we load safetensors)
```bash
df -h ~ | tail -1
CACHE=~/.cache/huggingface/hub/models--lmsys--vicuna-13b-v1.5-16k
for f in $(find "$CACHE" -name "*.bin"); do readlink -f "$f"; done | sort -u | xargs -r rm -f
find "$CACHE" -name "*.bin" -delete
df -h ~ | tail -1
```
Report free space before/after (need ~15+ GB free for Mistral).

## TASK 2 — Pull fixes + clear old smoke outputs
```bash
git pull --ff-only
rm -rf data/generated/C1-P0 data/generated/C2-P0 data/generated/C3-P0 data/generated/C4-P0
```

## TASK 3 — Re-smoke all four (GPU)
```bash
python generation/generate_c1.py --prompt P0 --n 2 --max-new-tokens 1024
python generation/generate_c2.py --prompt P0 --n 2 --max-turns 12
python generation/generate_c3.py --prompt P0 --n 2 --max-turns 12
python generation/generate_c4.py --prompt P0 --n 2 --max-turns 12
```
For EACH architecture, record in `VM_REPORT.md`:
- `n_turns`, `multi_turn_emissions` (C3/C4 may now be > 0 — that is the honest count)
- whether the assistant/help-desk framing is gone (quote 2–3 lines)
- the FULL `turns` list of ONE conversation per architecture (paste it — the local lead
  reads the raw text to judge quality)
- C4: did Vicuna + Mistral both fit in VRAM? peak VRAM? any OOM?

## TASK 4 — Hold
Commit + push, then hold. The local lead reviews the raw turns and decides whether to scale
to 50/condition or change the turn-by-turn approach.
