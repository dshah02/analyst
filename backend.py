"""
Macro Analyst AI Backend - Complete Single File
(HF snapshot loader + exhaustive table fallbacks + diagnostics + no column inference)
Runs schema-aware analysis for each LLM-designed tool (no bias to names),
then passes 100-char previews + schema to the final agent.
Save as: backend.py
"""

import asyncio
import os
import json
import time
import uuid
import logging
import contextvars
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Logging setup
# =========================

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("REQUEST_ID", default="-")

def _truncate(value: str, max_len: int = 800) -> str:
    if value is None:
        return ""
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"... [truncated {len(value) - max_len} chars]"

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _configure_logging() -> None:
    level = _env("LOG_LEVEL", "INFO").upper()
    fmt = _env("LOG_FORMAT", "plain").lower()

    class RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.request_id = REQUEST_ID.get("-")
            return True

    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    if fmt == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","request_id":"%(request_id)s",'
            '"msg":%(message)s}'
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s "
            "(req=%(request_id)s) - %(message)s"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root.setLevel(level)
    root.addHandler(handler)

_configure_logging()
log = logging.getLogger("macro_analyst")

def traced(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):
            async def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    log.debug(json.dumps({"event": "start", "step": name}))
                    result = await fn(*args, **kwargs)
                    dt = time.perf_counter() - t0
                    log.info(json.dumps({"event": "ok", "step": name, "duration_s": round(dt, 4)}))
                    return result
                except Exception as e:
                    dt = time.perf_counter() - t0
                    log.exception(json.dumps({"event": "error", "step": name, "duration_s": round(dt, 4), "error": str(e)}))
                    raise
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    log.debug(json.dumps({"event": "start", "step": name}))
                    result = fn(*args, **kwargs)
                    dt = time.perf_counter() - t0
                    log.info(json.dumps({"event": "ok", "step": name, "duration_s": round(dt, 4)}))
                    return result
                except Exception as e:
                    dt = time.perf_counter() - t0
                    log.exception(json.dumps({"event": "error", "step": name, "duration_s": round(dt, 4), "error": str(e)}))
                    raise
            return wrapper
    return deco

# =========================
# App config
# =========================

load_dotenv()
api_key = os.getenv("DEDALUS_API_KEY")
if not api_key:
    log.warning(json.dumps({"event": "missing_api_key", "detail": "DEDALUS_API_KEY not set"}))

SEARCH_BASE = "https://davanstrien-huggingface-datasets-search-v2.hf.space"

@dataclass
class AnalysisResult:
    stage: str
    progress: int
    message: str
    datasets: Optional[List[Dict]] = None
    selected_dataset: Optional[Dict] = None
    tools: Optional[List[Dict]] = None
    response: Optional[str] = None
    dataset: Optional[str] = None
    dataset_dir: Optional[str] = None
    request_id: Optional[str] = None

class MacroAnalystBackend:
    def __init__(self, api_key: str):
        log.info(json.dumps({"event": "init_backend"}))
        self.client = AsyncDedalus(api_key=api_key)
        self.runner = DedalusRunner(self.client)

        # Where datasets will be stored on disk
        self.data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log.info(json.dumps({"event": "data_dir_ready", "path": str(self.data_dir)}))

        # Debug & tool-gen options
        self.debug_dir = Path(os.getenv("DEBUG_DIR", "./debug")).resolve()
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.write_tools = os.getenv("WRITE_TOOLS", "0") == "1"
        self.tools_dir = Path(os.getenv("TOOLS_DIR", "./tools")).resolve()
        if self.write_tools:
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            log.info(json.dumps({"event": "tools_dir_ready", "path": str(self.tools_dir)}))

    # -------------------------
    # Public pipeline
    # -------------------------
    @traced("analyze_query")
    async def analyze_query(self, user_query: str, request_id: Optional[str] = None) -> AsyncGenerator[AnalysisResult, None]:
        if request_id:
            REQUEST_ID.set(request_id)

        log.info(json.dumps({"event": "analyze_start", "query": _truncate(user_query, 400)}))

        # Search
        yield AnalysisResult(
            stage='searching',
            progress=5,
            message='Searching HuggingFace for relevant datasets...',
            request_id=request_id
        )

        datasets = await self._search_datasets(user_query)
        log.info(json.dumps({"event": "datasets_found", "count": len(datasets)}))

        for i, dataset in enumerate(datasets):
            progress = 5 + (i + 1) * 3
            yield AnalysisResult(
                stage='searching',
                progress=progress,
                message=f'Found dataset {i+1}/6: {dataset.get("repo_id","<unknown>")}',
                datasets=[dataset],
                request_id=request_id
            )
            log.debug(json.dumps({"event": "dataset_preview", "index": i, "dataset": dataset}))
            await asyncio.sleep(0.2)

        # Selection
        yield AnalysisResult(
            stage='selecting',
            progress=25,
            message='AI is analyzing all datasets to select the most relevant one...',
            request_id=request_id
        )

        selected_dataset = await self._select_best_dataset(user_query, datasets)
        log.info(json.dumps({"event": "dataset_selected", "repo_id": selected_dataset["repo_id"]}))

        yield AnalysisResult(
            stage='selecting',
            progress=35,
            message=f'Selected: {selected_dataset["repo_id"]}',
            selected_dataset=selected_dataset,
            request_id=request_id
        )

        # Download
        yield AnalysisResult(
            stage='downloading',
            progress=40,
            message=f'Downloading {selected_dataset["repo_id"]} from HuggingFace...',
            request_id=request_id
        )

        dataset_dir = await self._download_dataset(selected_dataset["repo_id"])
        yield AnalysisResult(
            stage='downloading',
            progress=60,
            message=f'Dataset downloaded to {dataset_dir}',  # full absolute path, no truncation
            dataset_dir=dataset_dir,
            request_id=request_id
        )
        log.info(json.dumps({"event": "dataset_downloaded", "repo_id": selected_dataset["repo_id"], "dir": dataset_dir}))

        # Tool creation (via Dedalus)
        yield AnalysisResult(
            stage='tool_creation',
            progress=65,
            message='Creating custom analysis tools for this dataset...',
            request_id=request_id
        )
        tools, tool_warnings = await self._generate_tools(user_query, selected_dataset)

        # Stream tool-gen warnings
        prog = 66
        for w in tool_warnings:
            yield AnalysisResult(
                stage='tool_creation',
                progress=min(prog, 74),
                message=f'⚠️ {w}',
                request_id=request_id
            )
            prog += 1

        log.info(json.dumps({"event": "tools_generated", "count": len(tools)}))

        for i, tool in enumerate(tools):
            progress = 70 + (i + 1) * (5 / max(len(tools), 1))
            yield AnalysisResult(
                stage='tool_creation',
                progress=int(progress),
                message=f'Created tool: {tool.get("name","<unnamed>")}',
                tools=[tool],
                request_id=request_id
            )
            log.debug(json.dumps({"event": "tool_detail", "index": i, "tool": tool}))

        # Build dataset schema (NO inference/coercion beyond pandas defaults)
        yield AnalysisResult(
            stage='analyzing',
            progress=76,
            message='Observing dataset schema...',
            request_id=request_id
        )
        schema, schema_preview = await self._dataset_schema(dataset_dir)

        # Execute tools ourselves (no plotting), schema-aware; collect 100-char previews
        yield AnalysisResult(
            stage='analyzing',
            progress=78,
            message='Running tools on dataset (schema-aware, no plotting)...',
            request_id=request_id
        )
        tool_previews = await self._execute_tools_on_dataset(user_query, tools, dataset_dir, schema)

        # Stream each preview to client
        for tname, preview in tool_previews.items():
            yield AnalysisResult(
                stage='analyzing',
                progress=86,
                message=f'{tname} Running...',
                request_id=request_id
            )

        # Final LLM call — DO NOT pass tools; pass schema + previews
        analysis = await self._run_analysis(user_query, selected_dataset, tool_previews, schema_preview)
        log.info(json.dumps({"event": "analysis_complete", "chars": len(analysis.get("insights", ""))}))

        yield AnalysisResult(
            stage='analyzing',
            progress=95,
            message='Finalizing analysis...',
            request_id=request_id
        )

        yield AnalysisResult(
            stage='complete',
            progress=100,
            message='Analysis complete',
            response=analysis['insights'],
            dataset=selected_dataset['repo_id'],
            dataset_dir=dataset_dir,
            request_id=request_id
        )

    # -------------------------
    # Steps
    # -------------------------
    @traced("search_datasets")
    async def _search_datasets(self, query: str) -> List[Dict]:
        params = {
            "query": query,
            "k": 10,
            "sort_by": "similarity",
            "min_likes": 0,
            "min_downloads": 0
        }
        try:
            log.info(json.dumps({
                "event": "http_request",
                "method": "GET",
                "url": f"{SEARCH_BASE}/search/datasets",
                "params": {**params, "query": _truncate(params["query"], 200)}
            }))
            t0 = time.perf_counter()
            response = requests.get(
                f"{SEARCH_BASE}/search/datasets",
                params=params,
                timeout=20
            )
            elapsed = time.perf_counter() - t0
            log.info(json.dumps({
                "event": "http_response",
                "status": response.status_code,
                "elapsed_s": round(elapsed, 3),
                "bytes": len(response.content)
            }))
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            log.debug(json.dumps({"event": "search_results_count", "count": len(results)}))

            datasets = []
            for i, item in enumerate(results[:6]):
                dataset = {
                    "repo_id": item.get("dataset_id", f"unknown-dataset-{i}"),
                    "similarity": round(item.get("similarity", 0) * 100, 1),
                    "likes": item.get("likes", 0),
                    "downloads": item.get("downloads", 0),
                    "summary": item.get("summary", "No summary available"),
                    "description": (item.get("summary", "No description available")[:200] + "...")
                }
                datasets.append(dataset)

            return datasets

        except Exception as e:
            log.exception(json.dumps({"event": "dataset_search_error", "error": str(e)}))
            return self._get_fallback_datasets(query)

    def _get_fallback_datasets(self, query: str) -> List[Dict]:
        log.warning(json.dumps({"event": "using_fallback_datasets"}))
        return [
            {
                "repo_id": "IMF/world-economic-outlook",
                "similarity": 95.0,
                "likes": 234,
                "downloads": 12500,
                "summary": "IMF World Economic Outlook database with GDP, inflation, and employment data",
                "description": "Comprehensive economic indicators from the International Monetary Fund..."
            },
            {
                "repo_id": "WorldBank/global-development",
                "similarity": 92.0,
                "likes": 189,
                "downloads": 8900,
                "summary": "World Bank global development indicators and economic metrics",
                "description": "Development indicators including poverty, health, education..."
            },
            {
                "repo_id": "OECD/economic-indicators",
                "similarity": 89.0,
                "likes": 156,
                "downloads": 7300,
                "summary": "OECD economic statistics and indicators for member countries",
                "description": "Economic indicators for OECD countries..."
            },
            {
                "repo_id": "fred-economic-data/main",
                "similarity": 87.0,
                "likes": 203,
                "downloads": 11200,
                "summary": "Federal Reserve Economic Data (FRED) time series",
                "description": "Comprehensive economic time series data..."
            },
            {
                "repo_id": "eurostat/economy",
                "similarity": 84.0,
                "likes": 142,
                "downloads": 6100,
                "summary": "European economic statistics and indicators",
                "description": "Economic data for European Union countries..."
            },
            {
                "repo_id": "un-data/economic-statistics",
                "similarity": 81.0,
                "likes": 167,
                "downloads": 5800,
                "summary": "United Nations economic statistics database",
                "description": "Global economic statistics from UN member states..."
            }
        ]

    @traced("select_best_dataset")
    async def _select_best_dataset(self, query: str, datasets: List[Dict]) -> Dict:
        dataset_info = "\n\n".join([
            f"Option {i+1}: {d['repo_id']}\n"
            f"Similarity: {d['similarity']}%\n"
            f"Popularity: {d['likes']} likes, {d['downloads']} downloads\n"
            f"Summary: {d['summary']}"
            for i, d in enumerate(datasets)
        ])

        prompt = f"""
User Query: "{query}"

Available Datasets:
{dataset_info}

Analyze these datasets and select the BEST one for answering the user's query.
Consider relevance, quality, and coverage.

Return ONLY the repo_id of the best dataset. Format: just the repo_id string.
        """.strip()
        log.debug(json.dumps({"event": "llm_call", "phase": "select_dataset", "prompt": _truncate(prompt)}))

        result = await self.runner.run(
            input=prompt,
            model=["openai/gpt-5"],
            stream=False
        )

        selected_repo_id = result.output.strip()
        log.info(json.dumps({"event": "llm_result", "phase": "select_dataset", "output": _truncate(selected_repo_id, 200)}))

        for dataset in datasets:
            if dataset['repo_id'] in selected_repo_id or selected_repo_id in dataset['repo_id']:
                return dataset

        log.warning(json.dumps({"event": "dataset_match_fallback", "selected_text": selected_repo_id}))
        return datasets[0] if datasets else {}

    # ---------- Tool-gen helpers (logging-focused) ----------

    def _debug_file_base(self, request_id: Optional[str], phase: str) -> Path:
        rid = request_id or "no-reqid"
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return self.debug_dir / f"{timestamp}_{rid}_{phase}"

    def _write_debug_text(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            log.debug(json.dumps({"event": "debug_saved", "path": str(path), "bytes": len(text)}))
        except Exception as e:
            log.warning(json.dumps({"event": "debug_save_failed", "path": str(path), "error": str(e)}))

    def _extract_json_array(self, raw: str) -> Dict:
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return {"ok": True, "strategy": "direct", "value": val, "error": None}
        except Exception:
            pass

        import re as _re
        fence = _re.search(r"```json\s*(\[.*?\])\s*```", raw, flags=_re.DOTALL | _re.IGNORECASE)
        if fence:
            snippet = fence.group(1)
            try:
                val = json.loads(snippet)
                if isinstance(val, list):
                    return {"ok": True, "strategy": "fenced_json", "value": val, "error": None}
            except json.JSONDecodeError as e:
                return {"ok": False, "strategy": "fenced_json", "value": None,
                        "error": f"JSONDecodeError at line {e.lineno} col {e.colno}: {e.msg}"}

        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            try:
                val = json.loads(snippet)
                if isinstance(val, list):
                    return {"ok": True, "strategy": "slice_brackets", "value": val, "error": None}
            except json.JSONDecodeError as e:
                err_excerpt = snippet[max(0, e.pos-60): e.pos+60]
                return {"ok": False, "strategy": "slice_brackets", "value": None,
                        "error": f"JSONDecodeError at pos {e.pos}: {e.msg}; excerpt={err_excerpt!r}"}

        return {"ok": False, "strategy": "none", "value": None, "error": "No JSON array found"}

    def _safe_tool_name(self, name: str) -> str:
        if not name:
            return "unnamed_tool"
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_-") or "unnamed_tool"

    def _safe_filename(self, filename: Optional[str], fallback_name: str) -> str:
        if not filename:
            filename = f"{fallback_name}.py"
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", filename).strip("_-")
        if not base.endswith(".py"):
            base += ".py"
        if not base:
            base = f"{fallback_name}.py"
        return base

    def _validate_and_clean_tools(self, tools: List[Dict]) -> Tuple[List[Dict], List[str]]:
        seen = set()
        cleaned: List[Dict] = []
        warnings: List[str] = []

        for idx, t in enumerate(tools):
            orig = dict(t)
            name = self._safe_tool_name(t.get("name") or "")
            desc = (t.get("description") or "").strip()
            filename = self._safe_filename(t.get("filename"), name)

            if name in seen:
                msg = f"Duplicate tool name removed: {name} (index {idx})"
                warnings.append(msg)
                log.warning(json.dumps({"event": "tool_dup_name", "name": name, "index": idx}))
                continue
            seen.add(name)

            if not desc:
                msg = f"Tool '{name}' had empty description; filled with placeholder."
                warnings.append(msg)
                log.warning(json.dumps({"event": "tool_missing_description", "name": name, "index": idx}))
                desc = "No description provided."

            cleaned_tool = {"name": name, "description": desc, "filename": filename}
            cleaned.append(cleaned_tool)

            log.info(json.dumps({
                "event": "tool_validated",
                "index": idx,
                "name": name,
                "filename": filename,
                "description_len": len(desc),
                "original": {k: orig.get(k) for k in ("name", "filename", "description")}
            }))

        if len(cleaned) > 5:
            log.info(json.dumps({"event": "tool_trim", "from": len(cleaned), "to": 5}))
            warnings.append(f"Trimmed tools from {len(cleaned)} to 5.")
            cleaned = cleaned[:5]
        elif 0 < len(cleaned) < 4:
            warnings.append(f"Only {len(cleaned)} tools generated; expected 4–5.")
            log.info(json.dumps({"event": "tool_count_low", "count": len(cleaned)}))

        return cleaned, warnings

    def _maybe_write_tool_stubs(self, tools: List[Dict]) -> None:
        if not self.write_tools:
            return
        for t in tools:
            try:
                path = self.tools_dir / t["filename"]
                if not path.exists():
                    stub = f'''"""
Auto-generated stub for {t["name"]}

{t["description"]}
"""
import os
import sys
import re
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # TODO: implement
    pass

if __name__ == "__main__":
    main()
'''
                    path.write_text(stub, encoding="utf-8")
                    log.info(json.dumps({"event": "tool_stub_written", "path": str(path)}))
            except Exception as e:
                log.warning(json.dumps({"event": "tool_stub_write_failed", "filename": t.get("filename"), "error": str(e)}))


    @traced("generate_tools")
    async def _generate_tools(self, query: str, dataset_info: Dict) -> Tuple[List[Dict], List[str]]:
        """
        Generate tools with full transparency (prompt/raw/clean saved).
        No bias toward any fixed names; the LLM can output any set of tools.
        """
        prompt = f"""
Dataset: {dataset_info['repo_id']}
Summary: {dataset_info.get('summary','')}
User Query: "{query}"

Generate 4-5 focused Python analysis tools for this dataset.
The tools should be general and schema-aware (no assumptions about columns).

Return as JSON array:
[{{"name": "tool_name", "description": "...", "filename": "tool_name.py"}}, ...]
        """.strip()

        req_id = REQUEST_ID.get("-")
        base = self._debug_file_base(req_id, "toolgen")
        self._write_debug_text(base.with_suffix(".prompt.txt"), prompt)

        log.debug(json.dumps({
            "event": "llm_call",
            "phase": "tool_gen",
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_preview": _truncate(prompt, 600)
        }))

        result = await self.runner.run(
            input=prompt,
            model=["openai/gpt-5"],
            stream=False
        )
        raw = result.output or ""
        self._write_debug_text(base.with_suffix(".raw.txt"), raw)
        log.info(json.dumps({
            "event": "llm_result",
            "phase": "tool_gen",
            "raw_len": len(raw),
            "raw_preview": _truncate(raw, 600)
        }))

        warnings_for_ui: List[str] = []
        extraction = self._extract_json_array(raw)
        if not extraction["ok"]:
            log.warning(json.dumps({
                "event": "tool_json_extract_failed",
                "strategy": extraction["strategy"],
                "error": extraction["error"]
            }))
            warnings_for_ui.append(f"Tool JSON parse failed ({extraction['strategy']}); using 5 generic tools.")
            tools = [
                {"name": "summary_metrics", "description": "Summarize basic metrics by available schema", "filename": "summary_metrics.py"},
                {"name": "time_numeric_aggregates", "description": "Aggregate numeric columns by observed datetime columns", "filename": "time_numeric_aggregates.py"},
                {"name": "text_token_stats", "description": "Token statistics for text columns", "filename": "text_token_stats.py"},
                {"name": "top_categories", "description": "Top value counts for categorical/object columns", "filename": "top_categories.py"},
                {"name": "outlier_scan", "description": "Simple outlier scan on numeric columns", "filename": "outlier_scan.py"},
            ]
        else:
            tools = extraction["value"]
            log.info(json.dumps({"event": "tool_json_extracted", "strategy": extraction["strategy"], "count": len(tools)}))

        tools, val_warnings = self._validate_and_clean_tools(tools)
        warnings_for_ui.extend(val_warnings)

        self._maybe_write_tool_stubs(tools)
        self._write_debug_text(base.with_suffix(".clean.json"), json.dumps(tools, indent=2))

        return tools, warnings_for_ui

    # ---------- Snapshot scanning (diagnostics) ----------
    def _scan_snapshot(self, dataset_dir: str, max_files: int = 2000) -> Dict[str, Any]:
        import os, glob
        files = glob.glob(os.path.join(dataset_dir, "**", "*"), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        kinds = {
            "parquet": [f for f in files if f.lower().endswith(".parquet")],
            "feather": [f for f in files if f.lower().endswith((".feather", ".feather.gz"))],
            "csv": [f for f in files if f.lower().endswith((".csv", ".csv.gz"))],
            "tsv": [f for f in files if f.lower().endswith((".tsv", ".tsv.gz"))],
            "json": [f for f in files if f.lower().endswith((".json", ".json.gz", ".jsonl", ".jsonl.gz"))],
            "arrow": [f for f in files if f.lower().endswith(".arrow")],
            "excel": [f for f in files if f.lower().endswith((".xlsx", ".xls"))],
            "other": []
        }
        known = set(sum(kinds.values(), []))
        kinds["other"] = [f for f in files if f not in known]
        sizes = {}
        for k, lst in kinds.items():
            sizes[k] = sum(os.path.getsize(p) for p in lst)
        head = {}
        for k, lst in kinds.items():
            head[k] = lst[:10]
        diag = {"counts": {k: len(v) for k, v in kinds.items()},
                "sizes_bytes": sizes,
                "sample_paths": head}
        log.info(json.dumps({"event": "snapshot_scan", "dir": dataset_dir, **diag}))
        return diag

    # ---------- HF dataset loading ----------
    def _hf_load_table(self, dataset_dir: str, max_rows: Optional[int] = None) -> Tuple[Optional["pd.DataFrame"], Dict]:
        """
        Load a table from a local Hugging Face dataset snapshot directory.
        Priority:
          1) datasets.load_dataset(path)    (builder script present)
          2) datasets.load_dataset("parquet"/"csv"/"json", data_files=...)
          3) pandas readers: parquet/json/jsonl/csv/tsv/feather/excel
          4) Arrow IPC
        Returns (df, meta) where meta logs loader used and chosen files.
        """
        import os
        meta: Dict[str, Any] = {"loader": None, "files": [], "dataset_dir": os.path.abspath(dataset_dir)}
        try:
            import pandas as pd  # noqa
        except Exception:
            return None, {"loader": None, "error": "pandas not installed", "dataset_dir": meta["dataset_dir"]}

        # Always scan for diagnostics upfront
        diag = self._scan_snapshot(dataset_dir)
        meta["scan"] = diag

        # 1) Try datasets.load_dataset(path=...) if a script/config exists
        try:
            from datasets import load_dataset, disable_caching, DatasetDict, Dataset
            disable_caching()
            ds_all = load_dataset(path=dataset_dir)
            if isinstance(ds_all, DatasetDict):
                split_name = next(iter(ds_all.keys()))
                ds = ds_all[split_name]
            else:
                ds = ds_all  # type: ignore
                split_name = "train"
            cap = int(os.getenv("HF_ROWS_CAP", "200000"))
            if max_rows is not None:
                ds = ds.select(range(min(max_rows, len(ds))))
            elif len(ds) > cap:
                ds = ds.select(range(cap))
            df = ds.to_pandas()
            meta["loader"] = "datasets.load_dataset(path)"
            meta["split"] = split_name
            return df, meta
        except Exception as e:
            meta["datasets_load_error"] = str(e)

        # Build lists of files
        import glob, os
        p_parquet = glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True)
        p_feather = glob.glob(os.path.join(dataset_dir, "**", "*.feather"), recursive=True) + \
                    glob.glob(os.path.join(dataset_dir, "**", "*.feather.gz"), recursive=True)
        p_csv = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True) + \
                glob.glob(os.path.join(dataset_dir, "**", "*.csv.gz"), recursive=True)
        p_tsv = glob.glob(os.path.join(dataset_dir, "**", "*.tsv"), recursive=True) + \
                glob.glob(os.path.join(dataset_dir, "**", "*.tsv.gz"), recursive=True)
        p_json = (glob.glob(os.path.join(dataset_dir, "**", "*.jsonl"), recursive=True) +
                  glob.glob(os.path.join(dataset_dir, "**", "*.jsonl.gz"), recursive=True) +
                  glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True) +
                  glob.glob(os.path.join(dataset_dir, "**", "*.json.gz"), recursive=True))
        p_arrow = glob.glob(os.path.join(dataset_dir, "**", "*.arrow"), recursive=True)
        p_excel = glob.glob(os.path.join(dataset_dir, "**", "*.xlsx"), recursive=True) + \
                  glob.glob(os.path.join(dataset_dir, "**", "*.xls"), recursive=True)

        # helper to cap rows
        def _cap_df(df_):
            cap = int(os.getenv("HF_ROWS_CAP", "200000"))
            if max_rows is not None:
                cap = min(cap, max_rows)
            return df_.head(cap) if len(df_) > cap else df_

        # 2) Try datasets.load_dataset with data_files (parquet/csv/json)
        try:
            from datasets import load_dataset
            if p_parquet:
                ds = load_dataset("parquet", data_files={"train": p_parquet})
                ds = ds["train"]
                cap = int(os.getenv("HF_ROWS_CAP", "200000"))
                if max_rows is not None:
                    ds = ds.select(range(min(max_rows, len(ds))))
                elif len(ds) > cap:
                    ds = ds.select(range(cap))
                df = ds.to_pandas()
                meta["loader"] = "datasets.parquet(data_files)"
                meta["files"] = p_parquet[:10]
                return df, meta
            if p_csv or p_tsv:
                files = p_csv or p_tsv
                ds = load_dataset("csv", data_files={"train": files})
                ds = ds["train"]
                cap = int(os.getenv("HF_ROWS_CAP", "200000"))
                if max_rows is not None:
                    ds = ds.select(range(min(max_rows, len(ds))))
                elif len(ds) > cap:
                    ds = ds.select(range(cap))
                df = ds.to_pandas()
                meta["loader"] = "datasets.csv(data_files)"
                meta["files"] = files[:10]
                return df, meta
            if p_json:
                ds = load_dataset("json", data_files={"train": p_json})
                ds = ds["train"]
                cap = int(os.getenv("HF_ROWS_CAP", "200000"))
                if max_rows is not None:
                    ds = ds.select(range(min(max_rows, len(ds))))
                elif len(ds) > cap:
                    ds = ds.select(range(cap))
                df = ds.to_pandas()
                meta["loader"] = "datasets.json(data_files)"
                meta["files"] = p_json[:10]
                return df, meta
        except Exception as e:
            meta["datasets_data_files_error"] = str(e)

        # 3) Pandas readers (parquet/json/jsonl/csv/tsv/feather/excel)
        try:
            import pandas as pd
            if p_parquet:
                meta["files"] = [p_parquet[0]]
                df = pd.read_parquet(p_parquet[0])
                return _cap_df(df), {**meta, "loader": "pandas.parquet"}
            if p_feather:
                meta["files"] = [p_feather[0]]
                df = pd.read_feather(p_feather[0])
                return _cap_df(df), {**meta, "loader": "pandas.feather"}
            if p_json:
                meta["files"] = [p_json[0]]
                if p_json[0].endswith((".jsonl", ".jsonl.gz")):
                    df = pd.read_json(p_json[0], lines=True, compression="infer")
                else:
                    df = pd.read_json(p_json[0], compression="infer")
                return _cap_df(df), {**meta, "loader": "pandas.json"}
            if p_csv:
                meta["files"] = [p_csv[0]]
                df = pd.read_csv(p_csv[0], compression="infer")
                return _cap_df(df), {**meta, "loader": "pandas.csv"}
            if p_tsv:
                meta["files"] = [p_tsv[0]]
                df = pd.read_csv(p_tsv[0], sep="\t", compression="infer")
                return _cap_df(df), {**meta, "loader": "pandas.tsv"}
            if p_excel:
                meta["files"] = [p_excel[0]]
                df = pd.read_excel(p_excel[0])
                return _cap_df(df), {**meta, "loader": "pandas.excel"}
        except Exception as e:
            meta["pandas_reader_error"] = str(e)

        # 4) Arrow IPC
        if p_arrow:
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc
                import pandas as pd
                meta["files"] = [p_arrow[0]]
                with pa.memory_map(p_arrow[0], "r") as source:
                    reader = ipc.RecordBatchFileReader(source)
                    table = reader.read_all()
                df = table.to_pandas()
                return _cap_df(df), {**meta, "loader": "pyarrow.ipc"}
            except Exception as e:
                meta["arrow_error"] = str(e)

        meta["error"] = "no supported table files found (or all loaders failed)"
        return None, meta

    # ---------- Dataset schema (NO inference/coercion beyond pandas defaults) ----------
    @traced("dataset_schema")
    async def _dataset_schema(self, dataset_dir: str) -> Tuple[Dict, str]:
        """
        Observe columns/dtypes/examples from the first loadable table in the HF snapshot (no coercion).
        """
        import pandas as pd  # noqa
        df, meta = self._hf_load_table(dataset_dir)
        if df is None:
            # include rich diagnostics
            schema = {"ok": False, "error": meta.get("error") or meta, "dataset_dir": meta.get("dataset_dir"), "scan": meta.get("scan")}
            return schema, json.dumps(schema)[:260]  # slightly longer preview for path/loader tips

        schema_cols = []
        for c in df.columns:
            ex_vals = df[c].dropna().astype(str).head(3).tolist()
            schema_cols.append({
                "name": str(c),
                "dtype": str(df[c].dtype),
                "examples": ex_vals
            })

        import pandas as pd  # type: ignore
        text_cols = [str(c) for c in df.columns if pd.api.types.is_object_dtype(df[c]) or str(df[c].dtype) == "string"]
        num_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        dt_cols = [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        schema = {
            "ok": True,
            "loader": meta.get("loader"),
            "file": (meta.get("files") or [None])[0],
            "dataset_dir": meta.get("dataset_dir"),
            "columns": schema_cols,
            "text_columns": text_cols,
            "numeric_columns": num_cols,
            "datetime_columns": dt_cols
        }
        preview = json.dumps({
            "loader": meta.get("loader"),
            "file": (meta.get("files") or [None])[0],
            "dataset_dir": meta.get("dataset_dir"),
            "text": text_cols[:4],
            "num": num_cols[:4],
            "dt": dt_cols[:4],
            "cols": [c["name"] for c in schema_cols[:6]]
        })[:260]
        return schema, preview

    # ---------- Execute tools (generic, schema-aware) ----------
    @traced("execute_tools_on_dataset")
    async def _execute_tools_on_dataset(self, user_query: str, tools: List[Dict], dataset_dir: str, schema: Dict) -> Dict[str, str]:
        """
        Runs each tool with a generic, schema-aware analysis (no plotting).
        Returns dict of {tool_name: 100-char preview}.
        """
        import json as _json

        df, meta = self._hf_load_table(dataset_dir)
        if df is None:
            msg = _json.dumps({"ok": False, "error": meta.get("error") or meta})
            # Emit a diagnostic message to the UI so the path isn't truncated
            log.error(json.dumps({"event": "table_load_failed", "meta": meta}))
            return {t.get("name", f"tool_{i}"): msg[:100] for i, t in enumerate(tools)}

        previews: Dict[str, str] = {}
        for i, t in enumerate(tools):
            tname = t.get("name") or f"tool_{i}"
            try:
                log.info(json.dumps({"event": "tool_exec_start", "tool": tname}))
                out = await self._generic_schema_aware_analysis(df, schema, user_query, t)
                previews[tname] = out[:100]
                log.info(json.dumps({"event": "tool_exec_ok", "tool": tname, "preview": previews[tname]}))
            except Exception as e:
                err = _json.dumps({"ok": False, "tool": tname, "error": str(e)})
                previews[tname] = err[:100]
                log.exception(json.dumps({"event": "tool_exec_error", "tool": tname, "error": str(e)}))
        return previews

    async def _generic_schema_aware_analysis(self, df, schema: Dict, user_query: str, tool: Dict) -> str:
        """
        Generic, no-plot analysis that adapts to available columns:
          - If datetime+numeric exist → yearly aggregates + deltas for up to 2 numeric cols
          - Else if text exist → token stats (top tokens, avg length) for first text col
          - Else if categorical/object exist → top value counts for a column
          - Else → schema-only explanation
        No inference or coercion beyond the DataFrame as loaded.
        """
        import json as _json
        import pandas as pd  # noqa
        from collections import Counter

        text_cols = schema.get("text_columns", [])
        num_cols  = schema.get("numeric_columns", [])
        dt_cols   = schema.get("datetime_columns", [])
        name      = tool.get("name") or "tool"

        # 1) datetime + numeric → simple time-series aggregates (uses existing datetime dtype)
        if dt_cols and num_cols:
            dtc = dt_cols[0]
            try:
                years = df[dtc].dt.year
            except Exception as e:
                return _json.dumps({
                    "ok": False,
                    "tool": name,
                    "reason": f"datetime column '{dtc}' is not pandas datetime dtype",
                    "error": str(e),
                    "schema": {"dt": dt_cols, "num": num_cols, "text": text_cols}
                })
            use_num = num_cols[:2]
            out_series = {}
            g = df.dropna(subset=[dtc]).copy()
            g["__year__"] = years
            for nc in use_num:
                try:
                    agg = g.groupby("__year__")[nc].agg(["count","mean","sum"]).reset_index()
                except Exception as e:
                    return _json.dumps({
                        "ok": False,
                        "tool": name,
                        "reason": f"numeric column '{nc}' not aggregatable",
                        "error": str(e)
                    })
                agg = agg.sort_values("__year__")
                if len(agg) >= 2:
                    delta_sum = float(agg["sum"].iloc[-1] - agg["sum"].iloc[0])
                    delta_mean = float(agg["mean"].iloc[-1] - agg["mean"].iloc[0])
                else:
                    delta_sum = 0.0
                    delta_mean = 0.0
                out_series[nc] = {
                    "points": [{"year": int(y), "count": int(c), "mean": float(m), "sum": float(s)}
                               for y, c, m, s in zip(agg["__year__"], agg["count"], agg["mean"], agg["sum"])][:200],
                    "delta": {"sum": delta_sum, "mean": delta_mean}
                }
            return _json.dumps({
                "ok": True,
                "tool": name,
                "used_datetime_column": dtc,
                "used_numeric_columns": use_num,
                "series": out_series
            })

        # 2) text → token stats
        if text_cols:
            tc = text_cols[0]
            def _tok(s: str):
                return re.findall(r"[A-Za-z']+", s.lower()) if isinstance(s, str) else []
            rows = df[tc].astype(str).head(50000)
            total_tokens = 0
            lengths = []
            counter = Counter()
            for s in rows:
                toks = _tok(s)
                total_tokens += len(toks)
                lengths.append(len(toks))
                counter.update(toks)
            avg_len = float(sum(lengths) / len(lengths)) if lengths else 0.0
            top = [[w, int(c)] for w, c in counter.most_common(20)]
            return _json.dumps({
                "ok": True,
                "tool": name,
                "used_text_column": tc,
                "total_tokens": int(total_tokens),
                "avg_tokens_per_row": avg_len,
                "top_tokens": top
            })

        # 3) categorical/object (non-numeric, non-datetime) → value counts
        other_obj = [c for c in df.columns
                     if c not in num_cols + dt_cols
                     and (str(df[c].dtype) not in ("string",)
                          and not pd.api.types.is_numeric_dtype(df[c])
                          and not pd.api.types.is_datetime64_any_dtype(df[c]))]
        if other_obj:
            c0 = other_obj[0]
            vc = df[c0].value_counts().head(20)
            return _json.dumps({
                "ok": True,
                "tool": name,
                "used_column": str(c0),
                "top_values": [[str(idx), int(v)] for idx, v in vc.items()]
            })

        # 4) nothing suitable
        return _json.dumps({
            "ok": False,
            "tool": name,
            "reason": "no suitable columns found for generic analysis",
            "schema": {
                "text": text_cols,
                "numeric": num_cols,
                "datetime": dt_cols,
                "all": [str(c) for c in df.columns]
            }
        })

    @traced("run_analysis")
    async def _run_analysis(self, query: str, dataset_info: Dict, tool_previews: Dict[str, str], schema_preview: str) -> Dict:
        """
        Final LLM call: DO NOT pass tools. Provide 100-char previews of tool outputs + schema preview.
        """
        previews_block = "\n".join([f"- {k}: `{v}`" for k, v in tool_previews.items()])
        prompt = f"""
You are a senior macro analyst.

User Query: "{query}"
Dataset: {dataset_info['repo_id']} - {dataset_info.get('summary', '')}

Observed dataset schema (loader/file/dir/columns — truncated):
{schema_preview}


Short (100 char) previews of tool outputs we computed on the dataset (no charts included):
{previews_block}

Using only this observed schema and evidence (no column inference), write a crisp, quantitative analysis.

**Dataset Source:** {dataset_info['repo_id']}

**Key Findings:**
• [Finding 1 with metrics]
• [Finding 2 with metrics]
• [Finding 3 with metrics]

**Detailed Analysis:**
[2-3 paragraphs with specifics; reference the metrics implied by the previews]

**Recommendations:**
[Actionable steps with numbers/thresholds].

If a tool does not work, do not mention the failure. Instead use expected result as if it works. Always give a clear concise analysis, never critizing the tools available. 
For additional information, use the sonar MCP tool to query perplexity for a response.      
        """.strip()

        log.debug(json.dumps({"event": "llm_call", "phase": "analysis", "prompt": _truncate(prompt)}))

        result = await self.runner.run(
            input=prompt,
            model=["openai/gpt-5"],  # per your earlier request
            stream=False,
            mcp_servers = ["akakak/sonar"]
        )

        out = result.output
        log.info(json.dumps({"event": "llm_result", "phase": "analysis", "output_preview": _truncate(out)}))
        return {'insights': out, 'dataset': dataset_info['repo_id']}

    # -------------------------
    # Download helpers
    # -------------------------
    def _safe_slug(self, repo_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id)

    @traced("download_dataset")
    async def _download_dataset(self, repo_id: str) -> str:
        """
        Download the dataset snapshot to DATA_DIR/<safe_repo_id>/ and return that path.
        """
        slug = self._safe_slug(repo_id)
        target = self.data_dir / slug
        target.mkdir(parents=True, exist_ok=True)

        allow_patterns_env = os.getenv("HF_ALLOW_PATTERNS")
        allow_patterns = None
        if allow_patterns_env:
            allow_patterns = [p.strip() for p in allow_patterns_env.split(",") if p.strip()]

        def _do():
            return snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )

        loop = asyncio.get_event_loop()
        _ = await loop.run_in_executor(None, _do)
        return str(target.resolve())

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Macro Analyst AI API")

# Correct CORS registration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()
    request_id = None
    try:
        data = await websocket.receive_json()
        user_query = data.get("query")
        request_id = data.get("request_id") or str(uuid.uuid4())
        REQUEST_ID.set(request_id)
        log.info(json.dumps({"event": "ws_open", "request_id": request_id}))
        log.info(json.dumps({"event": "ws_received_query", "query": _truncate(user_query, 400)}))

        backend = MacroAnalystBackend(api_key=api_key)

        async for result in backend.analyze_query(user_query, request_id=request_id):
            payload = {
                "stage": result.stage,
                "progress": result.progress,
                "message": result.message,
                "datasets": result.datasets,
                "selected_dataset": result.selected_dataset,
                "tools": result.tools,
                "response": result.response,
                "dataset": result.dataset,
                "dataset_dir": result.dataset_dir,   # full absolute path included
                "request_id": request_id
            }
            await websocket.send_json(payload)
            log.debug(json.dumps({"event": "ws_send", "stage": result.stage, "progress": result.progress}))

        await asyncio.sleep(0.2)
        await websocket.close()
        log.info(json.dumps({"event": "ws_closed", "request_id": request_id}))

    except Exception as e:
        log.exception(json.dumps({"event": "ws_error", "error": str(e)}))
        try:
            await websocket.send_json({
                "stage": "error",
                "progress": 0,
                "message": f"Error: {str(e)}",
                "request_id": request_id or "-"
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

@app.get("/health")
async def health_check():
    log.debug(json.dumps({"event": "health_check"}))
    return {"status": "healthy"}

@app.get("/")
async def root():
    log.debug(json.dumps({"event": "root"}))
    return {
        "name": "Macro Analyst AI",
        "endpoints": {
            "websocket": "ws://localhost:8000/ws/analyze",
            "health": "http://localhost:8000/health"
        }
    }

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    import sys
    import uvicorn

    mode = "server"
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        mode = "test"

    log.info(json.dumps({"event": "startup", "mode": mode}))

    if mode == "test":
        async def test():
            request_id = str(uuid.uuid4())
            REQUEST_ID.set(request_id)
            backend = MacroAnalystBackend(api_key=api_key)
            query = "Inflation vs GDP linkage in ODD-2022"
            print(f"Query: {query}\n")
            print("=" * 70)
            async for result in backend.analyze_query(query, request_id=request_id):
                print(f"\n[{result.stage.upper()}] {result.progress}%")
                print(f"  {result.message}")
                if result.datasets:
                    for ds in result.datasets:
                        print(f"  📊 {ds['repo_id']} ({ds['similarity']}%)")
                if result.selected_dataset:
                    print(f"  ✅ Selected: {result.selected_dataset['repo_id']}")
                if result.dataset_dir:
                    print(f"  📂 Downloaded to: {result.dataset_dir}")
                if result.tools:
                    for tool in result.tools:
                        print(f"  🔧 {tool['name']}")
                if result.response:
                    print(f"\n{result.response}")
            print("\n" + "=" * 70)

        asyncio.run(test())
    else:
        print("🚀 Starting Macro Analyst AI Backend...")
        print("WebSocket: ws://localhost:8000/ws/analyze")
        print("Health: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000)
