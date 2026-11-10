#!/usr/bin/env bash
# Run every reviewer-checklist confound check against the CURRENT data/generated/
# conversations. Run this after every regeneration -- it's the single command that
# replaces re-doing each check by hand.
#
# Two local-only paths are required (neither is committed to the repo):
#   SB_ROOT   -- your local extracted Switchboard (swda/) directory. Switchboard is
#                LDC-licensed and must never be committed (see CLAUDE.md).
#   W2V_PATH  -- word2vec-google-news-300.gz (~1.7GB). Get it from
#                https://github.com/RaRe-Technologies/gensim-data
#                (release tag "word2vec-google-news-300").
#
# Usage:
#   SB_ROOT=/path/to/swda W2V_PATH=/path/to/word2vec-google-news-300.gz \
#       bash analysis/run_post_generation_checks.sh
#
# Dialogue-act tagging (step 0) needs the isolated DialogTag environment
# (analysis/requirements-dialogue-acts.txt -- legacy TensorFlow/Transformers pins) and is
# NOT run by this script automatically, since it shouldn't share an environment with the
# rest of this repo's dependencies. Run it yourself first, in that environment:
#   python analysis/dialogue_acts.py
# This script assumes that has already been done and results/dialogue_acts/ is fresh --
# step 4 below also needs that same isolated environment, since it calls DialogTag's own
# tag_both_sides() (reusing the prediction cache analysis/dialogue_acts.py just wrote, so
# it won't re-tag from scratch).

set -euo pipefail
cd "$(dirname "$0")/.."

: "${SB_ROOT:?Set SB_ROOT to your local swda/ directory}"
: "${W2V_PATH:?Set W2V_PATH to your word2vec-google-news-300.gz}"

echo "############################################################"
echo "# 1/4  Statement/Opinion confound (confusion table + merge + recompute + decide)"
echo "############################################################"
python3 analysis/check_statement_opinion_merge.py --all

echo
echo "############################################################"
echo "# 2/4  Backchannel standalone vs. turn-opener (unit-splitting confound)"
echo "############################################################"
python3 analysis/backchannel_standalone_vs_opener.py --sb-root "$SB_ROOT"

echo
echo "############################################################"
echo "# 3/4  Conceptual-alignment length confound (baseline, length control, decide)"
echo "############################################################"
python3 analysis/alignment_confound_check.py --sb-root "$SB_ROOT" --w2v-path "$W2V_PATH"

echo
echo "############################################################"
echo "# 4/4  Per-conversation distance: bootstrap error bars + architecture-vs-prompt ANOVA"
echo "############################################################"
python3 analysis/distance_bootstrap_stats.py

echo
echo "All checks complete."
