#!/usr/bin/env python3
"""Build the task-laptop <-> Elemem-host event-type crosswalk from raw logs.

For each System-4 experiment we scan the Elemem ``event.log`` files under
``/protocols/r1/subjects`` and take the union of the ``type`` field. The host
clock additionally carries Elemem-generated control/stim/classifier messages
that the task laptop never sends; the curated ``ANCHORS`` list keeps only the
behavioral types that appear on *both* clocks and can
anchor the non-heartbeat clock-drift correction in
``event_creation/.../alignment/system4.py:NONHB_EVENT_MAP``.

Outputs ``event_type_crosswalk_elemem_tasklaptop.csv`` and prints a dict literal
ready to paste into ``NONHB_EVENT_MAP``.
"""
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

PROTOCOLS = "/protocols/r1/subjects"
MAX_SESSIONS = 15  # sessions sampled per experiment
OUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "event_type_crosswalk_elemem_tasklaptop.csv",
)

# Per-experiment ordered anchor list (WORD-equivalent first). Restricts and
# orders the raw matched set into the alignment-anchor order we want in the dict.
ANCHORS = {
    "IFR1":    ["WORD", "ORIENT", "ENCODING", "COUNTDOWN", "DISTRACT", "RETRIEVAL", "TRIAL"],
    "IFR6":    ["WORD", "ORIENT", "ENCODING", "COUNTDOWN", "DISTRACT", "RETRIEVAL", "TRIAL"],
    "ICatFR1": ["WORD", "ORIENT", "ENCODING", "COUNTDOWN", "DISTRACT", "RETRIEVAL", "TRIAL"],
    "ICatFR6": ["WORD", "ORIENT", "ENCODING", "COUNTDOWN", "DISTRACT", "RETRIEVAL", "TRIAL"],
    "catFR1":  ["WORD", "ORIENT", "ENCODING", "COUNTDOWN", "DISTRACT", "RETRIEVAL", "TRIAL", "MATH"],
    "RepFR1":  ["WORD", "ISI", "RECALL", "COUNTDOWN", "TRIAL", "TRIALEND", "SESSION"],
    "RepFR2":  ["WORD", "ISI", "RECALL", "COUNTDOWN", "TRIAL", "TRIALEND", "SESSION", "READY"],
    "EFRCourierOpenLoop": ["OBJECT_PRESENTATION_BEGINS", "ORIENT", "ENCODING", "RETRIEVAL", "TRIAL", "OBJECT_RECALL_RECORDING_START", "CUED_RECALL_RECORDING_START"],
    "EFRCourierReadOnly": ["OBJECT_PRESENTATION_BEGINS", "ORIENT", "ENCODING", "RETRIEVAL", "TRIAL", "OBJECT_RECALL_RECORDING_START", "CUED_RECALL_RECORDING_START"],
    "CPS":     ["ENCODING", "TRIAL", "WAITING", "VOCALIZATION"],
    "OPS":     [],
}
WORD_ANCHOR = {"WORD", "OBJECT_PRESENTATION_BEGINS"}


def host_type_union(event_logs):
    counts = Counter()
    for ev in event_logs[:MAX_SESSIONS]:
        for line in open(ev, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                counts[json.loads(line).get("type")] += 1
            except (ValueError, TypeError):
                pass
    return counts


def main():
    logs = glob.glob(
        PROTOCOLS + "/*/experiments/*/sessions/*/behavioral/*/elemem/*/event.log"
    )
    by_exp = defaultdict(list)
    for ev in logs:
        exp = re.search(r"/experiments/([^/]+)/", ev).group(1)
        by_exp[exp].append(ev)

    rows = []
    dict_lines = []
    for exp in sorted(ANCHORS):
        host = host_type_union(by_exp.get(exp, []))
        # Validate each curated anchor actually appears in the host event.log
        # (grounds the dict in real logs). If no local logs exist for this
        # experiment, fall back to the curated list unfiltered.
        anchors = [a for a in ANCHORS[exp] if a in host or not by_exp.get(exp)]
        pairs = [(a, a) for a in anchors]
        dict_lines.append(
            "    %-22s %s," % ("'%s':" % exp, repr(pairs))
        )
        for task_t, host_t in pairs:
            rows.append({
                "experiment": exp,
                "task_type": task_t,
                "host_type": host_t,
                "is_word_anchor": task_t in WORD_ANCHOR,
                "notes": "primary onset anchor" if task_t in WORD_ANCHOR else "",
            })
        if not pairs:
            rows.append({
                "experiment": exp, "task_type": "", "host_type": "",
                "is_word_anchor": False,
                "notes": "no task behavioral message stream (stim-only)",
            })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["experiment", "task_type", "host_type",
                           "is_word_anchor", "notes"])
        w.writeheader()
        w.writerows(rows)

    print("Wrote %d rows to %s\n" % (len(rows), OUT_CSV))
    print("NONHB_EVENT_MAP = {")
    print("\n".join(dict_lines))
    print("}")


if __name__ == "__main__":
    main()
