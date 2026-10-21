"""High level functions: identify the compounds of a peak file."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from os import PathLike
from pathlib import Path

import pandas as pd

from alpinac.config import IdentificationConfig
from alpinac.identify import identify
from alpinac.io import read_peaks
from alpinac.peaks import split_compounds
from alpinac.results import IdentificationResult

logger = logging.getLogger(__name__)


def _identify_safe(
    peaks: pd.DataFrame, config: IdentificationConfig, label
) -> IdentificationResult:
    try:
        return identify(peaks, config, compound=label)
    except Exception as e:  # reported in the result, not raised
        logger.debug(traceback.format_exc())
        return IdentificationResult(
            compound=label,
            status="error",
            message=f"{type(e).__name__}: {e}",
            n_peaks=len(peaks),
        )


def _worker(conn, peaks, config, label):
    conn.send(_identify_safe(peaks, config, label))
    conn.close()


def identify_compounds(
    compounds: dict[int, pd.DataFrame],
    config: IdentificationConfig | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
) -> dict[int, IdentificationResult]:
    """Identify several compounds (peak tables by compound index).

    :param timeout: maximal time per compound in seconds (the compound is
        then reported with status ``"timeout"``). Uses worker processes.
    :param n_jobs: number of compounds identified in parallel.
    """
    config = config or IdentificationConfig()
    if timeout is None and n_jobs <= 1:
        return {k: _identify_safe(df, config, k) for k, df in compounds.items()}

    # fork: works from interactive sessions and scripts read from stdin, and
    # avoids re-importing (and re-compiling) everything in each worker.
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    results: dict[int, IdentificationResult] = {}
    pending = list(compounds.items())
    running: dict[int, tuple] = {}
    while pending or running:
        while pending and len(running) < max(n_jobs, 1):
            k, df = pending.pop(0)
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_worker, args=(send, df, config, k), daemon=True)
            proc.start()
            send.close()
            running[k] = (proc, recv, time.monotonic(), len(df))
        for k, (proc, recv, t0, n) in list(running.items()):
            if recv.poll():
                try:
                    results[k] = recv.recv()
                except EOFError:
                    results[k] = IdentificationResult(
                        compound=k, status="error", message="worker died", n_peaks=n
                    )
                proc.join()
                del running[k]
            elif not proc.is_alive():
                results[k] = IdentificationResult(
                    compound=k,
                    status="error",
                    message=f"worker exited ({proc.exitcode})",
                    n_peaks=n,
                )
                del running[k]
            elif timeout is not None and time.monotonic() - t0 > timeout:
                proc.terminate()
                proc.join()
                results[k] = IdentificationResult(
                    compound=k,
                    status="timeout",
                    message=f"took more than {timeout} s",
                    n_peaks=n,
                    runtime=time.monotonic() - t0,
                )
                del running[k]
        if running:
            time.sleep(0.01)
    return {k: results[k] for k in compounds}


def identify_file(
    path: str | PathLike,
    config: IdentificationConfig | None = None,
    compounds: list[int] | None = None,
    output_dir: str | PathLike | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
    **config_changes,
) -> dict[int, IdentificationResult]:
    """Identify the compounds of a peak file.

    :param path: a peak file (see :func:`alpinac.io.read_peaks`).
    :param config: parameters; keyword arguments override single fields,
        e.g. ``identify_file(path, target_elements="CHClF")``.
    :param compounds: only these compound indices (default: all, except -1).
    :param output_dir: if given, the results are saved there: one folder per
        compound (see :meth:`IdentificationResult.save`) and ``summary.csv``.
    :param timeout: maximal time per compound, seconds.
    :param n_jobs: compounds identified in parallel.
    :return: the results by compound index.
    """
    config = (config or IdentificationConfig()).replace(**config_changes)
    peaks = read_peaks(path)
    by_compound = split_compounds(peaks)
    if compounds is not None:
        missing = set(compounds) - set(by_compound)
        if missing:
            raise ValueError(f"Compounds {sorted(missing)} not in {path}")
        by_compound = {k: v for k, v in by_compound.items() if k in compounds}
    results = identify_compounds(by_compound, config, timeout=timeout, n_jobs=n_jobs)
    if output_dir is not None:
        save_results(results, output_dir, config)
    return results


def save_results(
    results: dict[int, IdentificationResult],
    output_dir: str | PathLike,
    config: IdentificationConfig | None = None,
) -> Path:
    """Save results: ``compound_<k>/`` folders, ``summary.csv`` and ``config.json``."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for k, r in results.items():
        path = r.save(output_dir / f"compound_{k}")
        logger.info("Saved results of compound %s in %s", k, path.resolve())
    summary = output_dir / "summary.csv"
    pd.DataFrame([r.summary() for r in results.values()]).to_csv(summary, index=False)
    logger.info("Saved summary in %s", summary.resolve())
    if config is not None:
        (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    return output_dir
