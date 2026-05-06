"""Heartbeat loading helpers for System-4 sessions.

HEARTBEATs are not stored in either the CML hd5 events or the BIDS tree —
both pipelines ultimately read them from ``/data10/RAM/subjects/.../session.jsonl``
(task laptop side) and ``/data10/.../elemem/*/event.log`` (host PC side).
This module exposes a single loader that returns the long-form DataFrame
expected by ``OnsetsHeartbeatsComparator`` and ``HeartbeatDelaysComparator``.

Ported from ``check_all_heartbeats.ipynb`` / ``fix_heartbeats_sys4.py``.
"""

from __future__ import annotations

import glob
import json
from typing import Optional

import numpy as np
import pandas as pd


_DATA10_ROOT = "/data10/RAM/subjects"


def _get_field(obj, key):
    """``obj[key]`` tolerating ``obj`` being a JSON-encoded string."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def _heart_one_side(
    subject: str,
    experiment: str,
    sess: int,
    *,
    subject_alias: str,
    load_host_pc: bool,
    drop_network_test: bool,
) -> pd.DataFrame:
    base = f"{_DATA10_ROOT}/{subject_alias}/behavioral/{experiment}/session_{sess}"
    if load_host_pc:
        pattern = f"{base}/elemem/*/event.log"
    else:
        pattern = (
            f"{base}/session.json"
            if experiment in ("catFR1",)
            else f"{base}/session.jsonl"
        )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one log for {subject}/{experiment}/ses-{sess}, "
            f"got {matches!r} from {pattern}"
        )

    df = _read_jsonl(matches[0])
    df["session"] = int(sess)

    if load_host_pc:
        df = df[df["type"].isin(["HEARTBEAT", "HEARTBEAT_OK"])]
        df["count"] = df["data"].apply(lambda x: _get_field(x, "count"))
    else:
        df["message"] = df["data"].apply(lambda x: _get_field(x, "message"))
        df = df.dropna(subset=["message"])
        df["type"] = df["message"].apply(lambda x: _get_field(x, "type"))
        df["data"] = df["message"].apply(lambda x: _get_field(x, "data"))
        df = df[df["type"].isin(["HEARTBEAT", "HEARTBEAT_OK"])]
        df["count"] = df["data"].apply(lambda x: _get_field(x, "count"))
        if len(df) == 0:
            raise ValueError(
                f"No HEARTBEAT events logged in {matches[0]} "
                f"({subject}/{experiment}/ses-{sess})"
            )

    if drop_network_test:
        df = df[df["count"] > 20]

    sent = df[df["type"] == "HEARTBEAT"].set_index("count", drop=False)
    done = df[df["type"] == "HEARTBEAT_OK"].set_index("count")

    latency = done["time"].astype(float) - sent["time"].astype(float)

    out = sent.copy()
    if "message" in out.columns:
        out = out.drop(columns=["message"])
    out["latency"] = latency
    out["time_HEARTBEAT_OK"] = done["time"]
    out["subject"] = subject
    out["experiment"] = experiment
    out["hardware_system"] = "host_pc" if load_host_pc else "task_laptop"
    out = out.reindex(columns=[
        "subject", "experiment", "session", "hardware_system",
        "count", "time", "time_HEARTBEAT_OK", "latency", "id",
    ])
    return out


def load_heartbeats_for_session(
    subject: str,
    experiment: str,
    session: int,
    *,
    subject_alias: Optional[str] = None,
    original_session: Optional[int] = None,
    drop_network_test: bool = True,
) -> pd.DataFrame:
    """Load HEARTBEAT/HEARTBEAT_OK rows for a single System-4 session.

    Parameters
    ----------
    subject, experiment, session
        Session identifiers as used by ``cmlreaders``. ``session`` is the
        cmlreaders session number; for System-4 sessions where the on-disk
        directory uses a different number, pass ``original_session``.
    subject_alias
        On-disk subject directory under ``/data10/RAM/subjects/`` (e.g.
        ``R1204T_1`` while ``subject == 'R1204T'``). Defaults to ``subject``.
    original_session
        Directory-name session number under
        ``/data10/RAM/subjects/<alias>/behavioral/<exp>/session_<sess>/``.
        Defaults to ``session``.
    drop_network_test
        If True, drop the initial network-test heartbeats (``count <= 20``).

    Returns
    -------
    pd.DataFrame
        Long-form heartbeats with rows for both ``hardware_system in
        {'task_laptop', 'host_pc'}``. Columns: ``subject, experiment,
        session, original_session, hardware_system, count, time,
        time_HEARTBEAT_OK, latency, id``.
    """
    alias = subject_alias if subject_alias else subject
    sess_dir = int(original_session if original_session is not None else session)

    task = _heart_one_side(
        subject, experiment, sess_dir,
        subject_alias=alias,
        load_host_pc=False, drop_network_test=drop_network_test,
    )
    host = _heart_one_side(
        subject, experiment, sess_dir,
        subject_alias=alias,
        load_host_pc=True, drop_network_test=drop_network_test,
    )
    out = pd.concat([task, host], ignore_index=True)
    out["session"] = int(session)
    out["original_session"] = sess_dir
    return out
