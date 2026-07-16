"""
Flexible vLLM version of run_gaia_workforce.py.

All 12 model call-sites in the GAIA workforce are routed through `make_model(role)`:

  Workforce orchestrators (3): coordinator, task, answerer
  Worker agents (3):           web, document, reasoning
  Toolkit-internal models (6): image, audio, browser_web, browser_planning,
                               video (CAMEL extracts frames client-side via
                                      BaseMessage.video_bytes, so any image-
                                      capable VL model works),
                               and document (reused for DocumentProcessingToolkit's
                                             long-doc re-ranking inside
                                             _post_process_result)

Of these 12 call-sites, only 3 roles actually feed images/video into the
LLM and therefore REQUIRE a vision-language model: `image`, `video`,
`browser_web` (SoM-annotated viewport screenshots). The other 9 roles only
ever see text — audio is transcribed by faster-whisper before reaching
the LLM, documents are parsed to text by pdfminer/etc. So the cheapest
split is 1 VL endpoint for {image, video, browser_web}.

Endpoint config lives in `gaia_workforce.yaml` (next to this script). Each
role under `roles:` gets `{ port, gpus }`; roles sharing the same `port`
point at the same vLLM server. The top-level `whisper_gpu: N` pins
faster-whisper's shared instance to GPU N. `gpus` per role is informational
(consumed by launch / profiling tooling); this script only reads `port`
and auto-discovers the served model name via /v1/models.

  GAIA_WORKFORCE_CONFIG=/path/to/other.yaml  → use a different config
  VLLM_HOST=remote-host                      → endpoint host (default localhost)
  GAIA_MAX_WORKERS=N (or `gaia N` argv)      → run N tasks concurrently
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from camel.models import ModelFactory
from camel.toolkits import (
    AsyncBrowserToolkit,
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    DocumentProcessingToolkit,
    ExcelToolkit,
    FunctionTool,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
)
from camel.tasks import Task
from camel.types import ModelPlatformType

from utils import OwlGaiaWorkforce, OwlWorkforceChatAgent
from utils.gaia import GAIABenchmark

load_dotenv(override=True)


# ──────────────────────────────────────────────────────────────────────────────
#                              vLLM endpoint config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VllmEndpoint:
    url: str
    api_key: str = "EMPTY"
    extra_config: Dict[str, Any] = field(default_factory=dict)


# All work products (tmp/, results/, data/, default config) are anchored to
# the script's directory so the script behaves identically regardless of CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
_TMP_DIR = _SCRIPT_DIR / "tmp"
# Results dir is overridable so a sweep can isolate each setting under its own
# <ts>-sweep/sweep_w<W>/ folder (CORAL-style) — separate folders mean no shared
# answer file accumulates, so a later setting can't "resume"/skip another's tasks,
# while every setting's results are preserved for review.
_RESULTS_DIR = Path(os.environ.get("GAIA_RESULTS_DIR") or (_SCRIPT_DIR / "results"))
_DATA_DIR = _SCRIPT_DIR / "data" / "gaia"

# Endpoint config is YAML-driven. Default path is `gaia_workforce.yaml`
# next to this script; override with `GAIA_WORKFORCE_CONFIG=<path>`.
_DEFAULT_CONFIG_PATH = _SCRIPT_DIR / "gaia_workforce.yaml"
_CONFIG_PATH = Path(os.environ.get("GAIA_WORKFORCE_CONFIG", _DEFAULT_CONFIG_PATH))
_HOST = os.environ.get("VLLM_HOST", "localhost")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Workforce config not found: {path}. Set GAIA_WORKFORCE_CONFIG "
            f"or place gaia_workforce.yaml next to this script."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config(_CONFIG_PATH)
logger.info(f"[vLLM] loaded workforce config from {_CONFIG_PATH}")

# whisper GPU pin (top-level scalar; None → faster-whisper default cuda:0)
_WHISPER_GPU_BASE: Optional[int] = _CONFIG.get("whisper_gpu")

# Build VLLM_ENDPOINTS from the YAML `roles` table. The yaml's `gpu` and
# `model` fields are informational (used by launch / profiling tooling, not
# this script) — the flex script only needs `port` to construct the URL,
# and auto-discovers the actual served model via /v1/models.
VLLM_ENDPOINTS: Dict[str, VllmEndpoint] = {}
for _role_name, _role_cfg in (_CONFIG.get("roles") or {}).items():
    _port = _role_cfg.get("port")
    if _port is None:
        continue
    VLLM_ENDPOINTS[_role_name] = VllmEndpoint(
        url=f"http://{_HOST}:{_port}/v1",
        api_key=_role_cfg.get("api_key", "EMPTY"),
        extra_config=_role_cfg.get("extra_config") or {},
    )

# Safety net: if no `default` is listed, alias the first role so
# _resolve_endpoint can always fall back for unknown role names.
if "default" not in VLLM_ENDPOINTS:
    if not VLLM_ENDPOINTS:
        raise ValueError(
            f"No usable vLLM roles in {_CONFIG_PATH} (need at least one role "
            f"with a non-null `port`)."
        )
    _first_role = next(iter(VLLM_ENDPOINTS))
    VLLM_ENDPOINTS["default"] = VLLM_ENDPOINTS[_first_role]
    logger.info(
        f"[vLLM] no `default` role in YAML; aliasing `{_first_role}` as default."
    )


# ──────────────────────────────────────────────────────────────────────────────
#                              model factory helper
# ──────────────────────────────────────────────────────────────────────────────

_model_name_cache: Dict[str, str] = {}


def _discover_model_name(url: str, api_key: str) -> str:
    if url in _model_name_cache:
        return _model_name_cache[url]
    cli = OpenAI(api_key=api_key, base_url=url)
    name = cli.models.list().data[0].id
    _model_name_cache[url] = name
    return name


def _resolve_endpoint(role: str) -> VllmEndpoint:
    if role in VLLM_ENDPOINTS:
        return VLLM_ENDPOINTS[role]
    return VLLM_ENDPOINTS["default"]


_GLOBAL_TOKEN_LIMIT: Optional[int] = _CONFIG.get("token_limit")


def _resolve_token_limit(role: str) -> Optional[int]:
    """Per-role override > global > None (camel falls back to model.token_limit)."""
    role_cfg = (_CONFIG.get("roles") or {}).get(role) or {}
    return role_cfg.get("token_limit") or _GLOBAL_TOKEN_LIMIT


def make_model(role: str, **extra_config):
    """Build a vLLM-backed CAMEL model for the given role."""
    ep = _resolve_endpoint(role)
    model_type = _discover_model_name(ep.url, ep.api_key)
    cfg: Dict[str, Any] = {"temperature": 0.5, **ep.extra_config, **extra_config}
    logger.info(f"[vLLM] role={role:<18s} url={ep.url:<32s} model={model_type}")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=model_type,
        model_config_dict=cfg,
        url=ep.url,
        api_key=ep.api_key,
    )
    model._agent_replay_role = role
    return model


# ──────────────────────────────────────────────────────────────────────────────
#                                workforce build
# ──────────────────────────────────────────────────────────────────────────────
#
# Whisper memory policy:
#   - lazy load: workers that never see an audio task never load whisper
#   - release-after-each: after each transcribe(), the model is dropped and
#     CTranslate2 actually returns the ~3.5GB to the OS (verified — unlike
#     PyTorch's caching allocator, CTranslate2 frees on destruction).
#     Costs ~2.6s reload per audio task but lets us run 64+ workers without
#     blowing up the whisper GPU. See _patch_transcribe_release_after_each.


def _patch_transcribe_release_after_each(toolkit) -> None:
    """Wrap toolkit._transcribe_local so the WhisperModel is freed back to
    the OS immediately after each transcription finishes. This caps the
    instantaneous whisper VRAM at "number of workers actively transcribing
    × 3.5GB" instead of "number of workers that have ever transcribed × 3.5GB".
    """
    import gc
    orig = toolkit._transcribe_local

    def _release():
        if toolkit._whisper_model is not None:
            del toolkit._whisper_model
            toolkit._whisper_model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def wrapped(audio_path):
        try:
            return orig(audio_path)
        finally:
            _release()

    toolkit._transcribe_local = wrapped

def construct_agent_list(worker_id: int = 0) -> List[Dict[str, Any]]:
    web_model                = make_model("web")
    document_processing_model = make_model("document")
    reasoning_model          = make_model("reasoning")
    image_analysis_model     = make_model("image")
    audio_reasoning_model    = make_model("audio")
    browser_web_model        = make_model("browser_web")
    browser_planning_model   = make_model("browser_planning")

    # Per-worker cache root so concurrent workforces don't trample each
    # other's downloaded docs / browser sessions / audio cache.
    tmp_root = str(_TMP_DIR / f"worker_{worker_id}")

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(
        cache_dir=tmp_root,
        image_analysis_model=image_analysis_model,
        text_processing_model=make_model("document"),
    )
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(
        download_directory=f"{tmp_root}/video",
        model=make_model("video"),
    )
    audio_analysis_toolkit = AudioAnalysisToolkit(
        cache_dir=f"{tmp_root}/audio",
        audio_reasoning_model=audio_reasoning_model,
        whisper_device_index=_WHISPER_GPU_BASE,
    )
    _patch_transcribe_release_after_each(audio_analysis_toolkit)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    browser_simulator_toolkit = AsyncBrowserToolkit(
        headless=True,
        cache_dir=f"{tmp_root}/browser",
        planning_agent_model=browser_planning_model,
        web_agent_model=browser_web_model,
    )
    excel_toolkit = ExcelToolkit()

    web_agent = OwlWorkforceChatAgent(
"""
You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.
- You can also simulate browser actions to get more information or verify the information you have found.
- Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
- Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
- In your response, you should mention the urls you have visited and processed.

Here are some tips that help you perform web search:
- Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
- If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
  For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
- The results you return do not have to directly answer the original question, you only need to collect relevant information.
""",
        model=web_model,
        token_limit=_resolve_token_limit("web"),
        tools=[
            FunctionTool(search_toolkit.search_duckduckgo),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.search_archived_webpage),
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(browser_simulator_toolkit.browse_url),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
        ],
    )
    web_agent._agent_replay_actor_id = "web"

    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        token_limit=_resolve_token_limit("document"),
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ],
    )
    document_processing_agent._agent_replay_actor_id = "document"

    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        token_limit=_resolve_token_limit("reasoning"),
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ],
    )
    reasoning_coding_agent._agent_replay_actor_id = "reasoning"

    return [
        {
            "name": "Web Agent",
            "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
            "agent": web_agent,
        },
        {
            "name": "Document Processing Agent",
            "description": "A helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc.",
            "agent": document_processing_agent,
        },
        {
            "name": "Reasoning Coding Agent",
            "description": "A helpful assistant that specializes in reasoning, coding, and processing excel files. However, it cannot access the internet to search for information. If the task requires python execution, it should be informed to execute the code after writing it.",
            "agent": reasoning_coding_agent,
        },
    ]


def construct_workforce(worker_id: int = 0) -> OwlGaiaWorkforce:
    coordinator_agent_kwargs = {"model": make_model("coordinator"),
                                "token_limit": _resolve_token_limit("coordinator")}
    task_agent_kwargs        = {"model": make_model("task"),
                                "token_limit": _resolve_token_limit("task")}
    answerer_agent_kwargs    = {"model": make_model("answerer"),
                                "token_limit": _resolve_token_limit("answerer")}

    workforce = OwlGaiaWorkforce(
        f"Gaia Workforce {worker_id}",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
        answerer_agent_kwargs=answerer_agent_kwargs,
    )

    for agent_dict in construct_agent_list(worker_id=worker_id):
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
        )
    return workforce


# ──────────────────────────────────────────────────────────────────────────────
#                                entry points
# ──────────────────────────────────────────────────────────────────────────────

def process_single_task(task_description: str, max_replanning_tries: int = 2) -> str:
    """Smoke test: run one ad-hoc question end-to-end."""
    task = Task(content=task_description)
    workforce = construct_workforce()
    processed = workforce.process_task(task, max_replanning_tries=max_replanning_tries)
    return workforce.get_workforce_final_answer(processed)


# ──────────────────────────────────────────────────────────────────────────────
#                            multi-process GAIA runner
# ──────────────────────────────────────────────────────────────────────────────
#
# Each worker is its own Python process (mp_context='spawn' — fork is unsafe
# once anything has touched CUDA in the parent). On startup `_proc_init`
# constructs a Workforce and binds it to a process-global, so subsequent
# tasks reuse it instead of paying the construction cost per-task.
#
# Result aggregation: workers `return` a result dict from `_proc_run_one`,
# the main process picks it up via `as_completed` and appends to the
# benchmark's `_results` (no IPC lock needed — only the main thread mutates).
#
# Memory lifecycle: every per-process global (workforce, agents, toolkits,
# any lazily-loaded whisper) lives until the worker process exits. The
# `with ProcessPoolExecutor(...)` context manager calls `shutdown(wait=True)`
# on exit, which terminates each worker — and that is when CUDA buffers
# (whisper weights, any cached allocator pool) are reclaimed by the OS.

_proc_workforce: Optional[OwlGaiaWorkforce] = None
_proc_benchmark: Optional[GAIABenchmark] = None
_proc_worker_id: Optional[int] = None


class _TaskTimeout(Exception):
    """Raised in a worker's main thread by SIGALRM when a single GAIA task
    exceeds its wall-clock budget (a task hung in a non-LLM tool op — e.g. a
    browser navigation/click — would otherwise occupy its concurrency slot
    forever, so steady-mode can never refill it and the offered load decays)."""


def _task_alarm_handler(signum, frame):
    raise _TaskTimeout()


def _proc_init(data_dir: str, save_to: str) -> None:
    """Per-worker-process initializer. Builds the Workforce once and stashes
    it on a module global so every task this worker runs reuses it."""
    global _proc_workforce, _proc_benchmark, _proc_worker_id
    _proc_worker_id = os.getpid()
    # Per-task watchdog: SIGALRM fires in this worker's main thread (where the
    # submitted callable runs), interrupting a hung task. No-op if unsupported.
    try:
        signal.signal(signal.SIGALRM, _task_alarm_handler)
    except (ValueError, OSError):
        pass
    _proc_benchmark = GAIABenchmark(data_dir=data_dir, save_to=save_to)
    # _proc_benchmark.load()
    _proc_workforce = construct_workforce(worker_id=_proc_worker_id)
    logger.info(f"[worker pid={_proc_worker_id}] initialized")


def _proc_run_one_impl(
    task: Dict[str, Any],
    max_tries: int,
    max_replanning_tries: int,
) -> Optional[Dict[str, Any]]:
    """Run one GAIA task to completion (with retries) in this worker process.
    Returns the result dict to send back to the main process, or None if the
    task could not be prepared (e.g. missing input file)."""
    assert _proc_workforce is not None and _proc_benchmark is not None
    wf = _proc_workforce
    bench = _proc_benchmark

    # Stamp this worker's current task so the t2t latency sidecar
    # (OpenAICompatibleModel._record_t2t) can attribute each LLM call to a GAIA
    # task under ProcessPoolExecutor concurrency. A worker runs one task at a
    # time, so this env var is unambiguous for the task's whole duration.
    os.environ["OWL_T2T_TASK"] = str(task.get("task_id", ""))

    success = False
    tries = 0
    trajectory_with_retry: List[dict] = []
    final_result: Optional[Dict[str, Any]] = None
    _task_to = int(os.environ.get("GAIA_TASK_TIMEOUT_S", "0") or "0")

    while not success and tries < max_tries:
        tries += 1
        logger.info(
            f"Attempt {tries}/{max_tries} for task {task['task_id']} "
            f"(worker pid={_proc_worker_id})"
        )
        try:
            if _task_to > 0:
                signal.alarm(_task_to)   # watchdog for this attempt
            valid, error_msg = bench._prepare_task(task)
            if not valid:
                logger.error(error_msg)
                break
            logger.info(f"Task Question: {task['Question']}")
            camel_task = bench._create_task(task)
            if wf.is_running():
                wf.stop()
            processed_task = wf.process_task(
                camel_task, max_replanning_tries=max_replanning_tries
            )

            try:
                answer = wf.get_workforce_final_answer(processed_task)
            except Exception as e:
                logger.error(f"Error extracting final answer: {e}")
                answer = None

            logger.info(
                f"Model answer: {answer}, Ground truth: {task['Final answer']}"
            )
            score = bench.question_scorer(answer, task["Final answer"])
            logger.info(f"Score: {score}")
            success = bool(score)
            trajectory_with_retry.append({
                "attempts": tries,
                "model_answer": answer,
                "ground_truth": task["Final answer"],
                "success": success,
                "trajectory": wf.get_overall_task_solve_trajectory(),
            })

            if success or tries == max_tries:
                final_result = {
                    "task_id": task["task_id"],
                    "question": task["Question"],
                    "level": task["Level"],
                    "model_answer": answer,
                    "ground_truth": task["Final answer"],
                    "score": score,
                    "attempts": tries,
                    "trajectory": trajectory_with_retry,
                }
        except _TaskTimeout:
            # hung task — abort this attempt, don't retry, free the slot so a
            # steady-mode refill can keep concurrency pinned.
            logger.error(
                f"[timeout] task {task['task_id']} exceeded {_task_to}s "
                f"(worker pid={_proc_worker_id}); aborting to free the slot"
            )
            try:
                if wf.is_running():
                    wf.stop()
            except Exception:
                pass
            final_result = {
                "task_id": task["task_id"],
                "question": task["Question"],
                "level": task["Level"],
                "model_answer": None,
                "ground_truth": task["Final answer"],
                "score": False,
                "attempts": tries,
                "trajectory": trajectory_with_retry,
                "timed_out": True,
            }
            break
        except Exception as e:
            logger.error(f"Error in processing task (attempt {tries}): {e}")
            if tries == max_tries:
                final_result = {
                    "task_id": task["task_id"],
                    "question": task["Question"],
                    "level": task["Level"],
                    "model_answer": None,
                    "ground_truth": task["Final answer"],
                    "score": False,
                    "attempts": tries,
                    "trajectory": trajectory_with_retry,
                }
        finally:
            if _task_to > 0:
                signal.alarm(0)   # disarm before next attempt / return

    capture_dir = os.environ.get("AGENT_REPLAY_OWL_CAPTURE_DIR")
    if capture_dir and final_result is not None:
        result_path = Path(capture_dir) / f"task-result.{task['task_id']}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(f"{result_path.suffix}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(final_result, default=str, indent=2) + "\n")
        os.replace(temporary, result_path)
    return final_result


def _proc_run_one(
    task: Dict[str, Any],
    max_tries: int,
    max_replanning_tries: int,
) -> Optional[Dict[str, Any]]:
    os.environ["AGENT_REPLAY_OWL_TASK_ID"] = str(task["task_id"])
    original_cwd = os.getcwd()
    try:
        # Model-generated execute_code snippets frequently create relative-path
        # scratch files.  Keep those files available across all tool calls for
        # this task, but isolate concurrent tasks and remove their scratch data
        # when the task ends instead of polluting frameworks/owl.
        task_prefix = str(task["task_id"]).replace(os.sep, "_")[:16]
        with tempfile.TemporaryDirectory(prefix=f"owl-gaia-{task_prefix}-") as workdir:
            os.chdir(workdir)
            try:
                return _proc_run_one_impl(task, max_tries, max_replanning_tries)
            finally:
                os.chdir(original_cwd)
    finally:
        os.environ.pop("AGENT_REPLAY_OWL_TASK_ID", None)


def _run_gaia_parallel(
    benchmark: GAIABenchmark,
    on: str,
    level: int,
    max_tries: int,
    max_replanning_tries: int,
    save_result: bool,
    max_workers: int,
    subset: Optional[int] = None,
    test_idx: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Multi-process GAIA runner. Spawns N persistent worker processes via
    ProcessPoolExecutor; each worker constructs its own Workforce in
    `_proc_init` and reuses it across many tasks. The main process owns
    `benchmark._results` and the save file, so no IPC lock is needed.
    """
    tasks = benchmark._load_tasks(on, level, randomize=False, subset=subset, idx=test_idx)

    benchmark._results = []
    if save_result:
        benchmark._results = benchmark._load_results_from_file(benchmark.save_to)

    pending: List[Dict[str, Any]] = []
    for t in tasks:
        if benchmark._check_task_completed(t["task_id"]):
            logger.success(
                f"The following task is already completed:\n "
                f"task id: {t['task_id']}, question: {t['Question']}"
            )
        else:
            pending.append(t)

    if not pending:
        return benchmark._generate_summary()

    ctx = mp.get_context("spawn")
    from tqdm import tqdm
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_proc_init,
        initargs=(str(benchmark.data_dir), benchmark.save_to),
    ) as ex:
        futures = [
            ex.submit(_proc_run_one, t, max_tries, max_replanning_tries)
            for t in pending
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Running {on} set (×{max_workers})",
        ):
            try:
                result = fut.result()
            except Exception as e:
                logger.error(f"Worker raised unhandled exception: {e}")
                continue
            if result is None:
                continue
            benchmark._results.append(result)
            if save_result:
                benchmark._save_results_to_file(benchmark._results, benchmark.save_to)

    return benchmark._generate_summary()


def _run_gaia_steady(
    benchmark: GAIABenchmark,
    on: str,
    level: int,
    max_tries: int,
    max_replanning_tries: int,
    max_workers: int,
    wall_s: float,
    subset: Optional[int] = None,
    test_idx: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Steady-concurrency LOAD runner: hold exactly `max_workers` GAIA tasks
    in-flight at all times by refilling from a seeded, *cycled* task queue, until
    `wall_s` seconds elapse (then stop refilling and drain).

    Unlike `_run_gaia_parallel` (fire the sampled set once -> concurrency decays
    as tasks finish), this keeps concurrency pinned at W for the whole window by
    re-submitting a new task the instant one completes, cycling the sampled tasks
    when they run out. Because tasks are REPEATED, this is a load-generation mode:
    completed task_ids are NOT skipped and results are NOT saved.
    """
    tasks = benchmark._load_tasks(on, level, randomize=False, subset=subset, idx=test_idx)
    if not tasks:
        logger.warning("[steady] no tasks resolved; nothing to run")
        return {"correct": 0, "total": 0, "accuracy": 0.0}

    from itertools import cycle as _cycle
    queue = _cycle(tasks)
    deadline = time.monotonic() + wall_s
    ctx = mp.get_context("spawn")
    correct = total = 0
    seen: Dict[str, Any] = {}   # first result per task_id, saved for review
    last_log = time.monotonic()
    events_path = os.environ.get("GAIA_QUEUE_EVENTS_PATH")
    refill_count = 0

    def queue_event(kind: str, **fields: Any) -> None:
        if not events_path:
            return
        record = {
            "event": kind,
            "ts_epoch": time.time(),
            "target_concurrency": max_workers,
            **fields,
        }
        path = Path(events_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    logger.info(
        f"[steady] pinning {max_workers} concurrent workforce(s) over "
        f"{len(tasks)} sampled task(s) (cycled) for {wall_s:.0f}s"
    )
    queue_event("queue_start", running=0, duration_s=wall_s)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_proc_init,
        initargs=(str(benchmark.data_dir), benchmark.save_to),
    ) as ex:
        inflight = {ex.submit(_proc_run_one, next(queue), max_tries, max_replanning_tries)
                    for _ in range(max_workers)}
        queue_event("queue_filled", running=len(inflight), refill_count=refill_count)
        while inflight:
            done, inflight = wait(inflight, timeout=10, return_when=FIRST_COMPLETED)
            refill = time.monotonic() < deadline
            for fut in done:
                try:
                    r = fut.result()
                    if r is not None:
                        total += 1
                        if r.get("score"):
                            correct += 1
                        # preserve the first result per unique task for review
                        # (repeats are load-only); save to the per-setting file.
                        tid = r.get("task_id")
                        if tid is not None and tid not in seen:
                            seen[tid] = r
                            benchmark._save_results_to_file(
                                list(seen.values()), benchmark.save_to)
                except Exception as e:
                    logger.error(f"[steady] worker exception: {e}")
                queue_event("task_end", running=len(inflight), refill_count=refill_count)
                if refill:
                    inflight.add(ex.submit(_proc_run_one, next(queue),
                                           max_tries, max_replanning_tries))
                    refill_count += 1
                    queue_event("refill", running=len(inflight), refill_count=refill_count)
            if time.monotonic() - last_log >= 30:
                phase = "filling" if refill else "draining"
                logger.info(f"[steady] {phase}: {len(inflight)} in-flight, "
                            f"{total} task-runs done ({correct} correct)")
                queue_event(
                    "heartbeat",
                    running=len(inflight),
                    refill_count=refill_count,
                    phase=phase,
                    completed=total,
                )
                last_log = time.monotonic()
    queue_event("queue_stop", running=0, refill_count=refill_count, completed=total)
    acc = correct / total if total else 0.0
    logger.success(f"[steady] done: {total} task-runs, {correct} correct, acc={acc:.3f}")
    return {"correct": correct, "total": total, "accuracy": acc}


def _run_gaia_rps(
    benchmark: GAIABenchmark,
    on: str,
    level: int,
    max_tries: int,
    max_replanning_tries: int,
    max_workers: int,
    wall_s: float,
    rps: float,
    seed: int = 0,
    subset: Optional[int] = None,
    test_idx: Optional[List[int]] = None,
    cycle: bool = False,
) -> Dict[str, Any]:
    """Open-loop OPEN-ARRIVAL runner: submit GAIA tasks as a Poisson(rps) process
    (exponential inter-arrival times), then drain the in-flight tasks.

    Task supply (mirrors sweep.sh's finite, seeded, shuffle-then-prefix sampling —
    the sampled --idx set is passed in via test_idx):
      * cycle=False (default): submit the resolved FINITE task set exactly ONCE,
        then stop. Total submissions == len(tasks) <= dataset size, so there is
        no unbounded queue buildup. `wall_s` is only an upper time bound (stop
        submitting early if it is reached first).
      * cycle=True: repeat the set indefinitely (load-generation) until `wall_s`,
        like steady mode. Only then can submissions exceed the dataset size.

    Contrast with `_run_gaia_steady` (closed-loop: pin exactly `max_workers`
    in-flight). Here `max_workers` is only a CAP: arrivals are driven by `rps`,
    so realized concurrency is an emergent random variable ~ rps * task_time
    (Little's law), bounded above by `max_workers`. If rps*task_time approaches
    `max_workers` the pool saturates and the effective start-rate falls below
    `rps` — surfaced as a "backlog" warning. Does NOT skip/save by task_id; the
    first result per unique task is saved for review.
    """
    tasks = list(benchmark._load_tasks(on, level, randomize=False, subset=subset, idx=test_idx))
    if not tasks:
        logger.warning("[rps] no tasks resolved; nothing to run")
        return {"correct": 0, "total": 0, "accuracy": 0.0}
    if rps <= 0:
        logger.warning("[rps] rps must be > 0; nothing to run")
        return {"correct": 0, "total": 0, "accuracy": 0.0}

    # Seeded shuffle of the resolved pool (mirrors sweep.sh's shuffle-then-prefix
    # sampling): the arrival stream is a reproducible random permutation of the
    # whole set, so mixed levels/durations are spread over time rather than
    # clustered by level order. Uses a dedicated RNG so it doesn't perturb the
    # arrival-timing RNG below (both seeded => fully reproducible).
    random.Random(seed).shuffle(tasks)

    from itertools import cycle as _cycle
    queue = _cycle(tasks) if cycle else iter(tasks)
    ctx = mp.get_context("spawn")
    rng = random.Random(seed)
    inflight: set = set()
    lock = threading.Lock()
    submit_done = threading.Event()
    submitted = 0
    correct = total = 0
    seen: Dict[str, Any] = {}
    supply = ("cycled/unbounded" if cycle
              else f"finite: {len(tasks)} task(s), submitted once")
    logger.info(
        f"[rps] Poisson arrivals at {rps:.3g} task/s (seed={seed}); "
        f"supply={supply}; wall<={wall_s:.0f}s; pool cap max_workers={max_workers}; "
        f"expected steady in-flight ~= rps*task_time (keep < cap to hold the rate)"
    )

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_proc_init,
        initargs=(str(benchmark.data_dir), benchmark.save_to),
    ) as ex:

        def _submitter():
            nonlocal submitted
            deadline = time.monotonic() + wall_s
            while True:
                # exponential inter-arrival => Poisson process at rate `rps`
                wait_s = rng.expovariate(rps)
                if time.monotonic() + wait_s >= deadline:
                    logger.info("[rps] wall reached; stop submitting")
                    break
                time.sleep(wait_s)
                try:
                    task = next(queue)      # finite iter exhausts (cycle=False)
                except StopIteration:
                    logger.info(f"[rps] finite task set exhausted after "
                                f"{submitted} submission(s); stop submitting")
                    break
                fut = ex.submit(_proc_run_one, task,
                                max_tries, max_replanning_tries)
                with lock:
                    inflight.add(fut)
                    submitted += 1
            submit_done.set()

        submitter = threading.Thread(target=_submitter, name="rps-submitter", daemon=True)
        submitter.start()

        last_log = time.monotonic()
        while True:
            with lock:
                current = list(inflight)
            if not current:
                if submit_done.is_set():
                    break
                time.sleep(0.2)
                continue
            done, _ = wait(current, timeout=5, return_when=FIRST_COMPLETED)
            for fut in done:
                with lock:
                    inflight.discard(fut)
                try:
                    r = fut.result()
                except Exception as e:
                    logger.error(f"[rps] worker exception: {e}")
                    continue
                if r is None:
                    continue
                total += 1
                if r.get("score"):
                    correct += 1
                tid = r.get("task_id")
                if tid is not None and tid not in seen:
                    seen[tid] = r
                    benchmark._save_results_to_file(list(seen.values()), benchmark.save_to)
            if time.monotonic() - last_log >= 30:
                with lock:
                    n_inflight = len(inflight)
                # futures beyond the pool size are queued inside the executor,
                # i.e. arrivals that could not start => the offered rps exceeds
                # what `max_workers` can serve at this task duration.
                backlog = max(0, n_inflight - max_workers)
                phase = "arriving" if not submit_done.is_set() else "draining"
                msg = (f"[rps] {phase}: {n_inflight} in-flight "
                       f"(cap {max_workers}), {submitted} submitted, "
                       f"{total} done ({correct} correct)")
                if backlog > 0:
                    logger.warning(msg + f"; BACKLOG {backlog} queued — "
                                   f"rps too high for max_workers, rate not held")
                else:
                    logger.info(msg)
                last_log = time.monotonic()

    acc = correct / total if total else 0.0
    logger.success(f"[rps] done: {submitted} submitted, {total} task-runs, "
                   f"{correct} correct, acc={acc:.3f}")
    return {"correct": correct, "total": total, "accuracy": acc}


def evaluate_on_gaia(level: int = 1, on: str = "valid", max_tries: int = 3,
                     test_idx: Optional[List[int]] = None,
                     save_result: bool = True,
                     max_workers: int = 1,
                     subset: Optional[int] = None,
                     max_replanning_tries: int = 2,
                     steady: bool = False,
                     steady_wall_s: float = 0.0,
                     rps: float = 0.0,
                     rps_wall_s: float = 0.0,
                     rps_seed: int = 0,
                     rps_cycle: bool = False):
    """Full GAIA benchmark sweep.

    Args:
        max_workers: Number of GAIA tasks to run concurrently. Each worker
            builds its own Workforce (with isolated tmp/worker_<id>/ caches),
            and the underlying vLLM endpoints multiplex the requests.
            ``max_workers == 1`` falls back to the stock sequential
            ``benchmark.run_workforce_with_retry`` path (zero behavior change).
    """
    if test_idx is None and subset is None:
        test_idx = [1]

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(_RESULTS_DIR / f"workforce_{level}_pass{max_tries}_vllm.json")
    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR)

    benchmark = GAIABenchmark(data_dir=str(_DATA_DIR), save_to=save_path)

    if rps > 0:
        # open-loop Poisson-arrival load mode: submit at `rps` task/s for
        # rps_wall_s, then drain. max_workers is only a cap. No skip/save by id.
        result = _run_gaia_rps(
            benchmark,
            on=on,
            level=level,
            test_idx=test_idx,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
            max_workers=max_workers,
            wall_s=rps_wall_s,
            rps=rps,
            seed=rps_seed,
            subset=subset,
            cycle=rps_cycle,
        )
    elif steady:
        # steady-concurrency load mode (refilling cycled queue) — any W, no save.
        result = _run_gaia_steady(
            benchmark,
            on=on,
            level=level,
            test_idx=test_idx,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
            max_workers=max_workers,
            wall_s=steady_wall_s,
            subset=subset,
        )
    elif max_workers <= 1 and not os.environ.get("AGENT_REPLAY_OWL_CAPTURE_DIR"):
        workforce = construct_workforce(worker_id=0)
        result = benchmark.run_workforce_with_retry(
            workforce,
            on=on,
            level=level,
            idx=test_idx,
            subset=subset,
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
        )
    else:
        result = _run_gaia_parallel(
            benchmark,
            on=on,
            level=level,
            test_idx=test_idx,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
            save_result=save_result,
            max_workers=max_workers,
            subset=subset,
        )

    logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "gaia":
        # CLI: `python run_gaia_workforce_vllm_flex.py gaia [max_workers] [subset]`
        # Env knobs (override evaluate_on_gaia defaults):
        #   GAIA_MAX_WORKERS, GAIA_SUBSET, GAIA_LEVEL (int or "all"),
        #   GAIA_ON (valid|test), GAIA_MAX_TRIES, GAIA_MAX_REPLANNING_TRIES,
        #   GAIA_TEST_IDX (comma list of ints, e.g. "0,5,12")
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            max_workers = int(sys.argv[2])
        else:
            max_workers = int(os.environ.get("GAIA_MAX_WORKERS", "1"))
        if len(sys.argv) > 3 and sys.argv[3].isdigit():
            subset = int(sys.argv[3])
        else:
            _env_subset = os.environ.get("GAIA_SUBSET")
            subset = int(_env_subset) if _env_subset else None

        kwargs: Dict[str, Any] = {}
        if (_v := os.environ.get("GAIA_LEVEL")):
            # int ("1"), list ("1,2"), or "all"
            if "," in _v:
                kwargs["level"] = [int(x) for x in _v.split(",") if x.strip()]
            elif _v.isdigit():
                kwargs["level"] = int(_v)
            else:
                kwargs["level"] = _v  # "all"
        if (_v := os.environ.get("GAIA_ON")):
            kwargs["on"] = _v
        if (_v := os.environ.get("GAIA_MAX_TRIES")):
            kwargs["max_tries"] = int(_v)
        if (_v := os.environ.get("GAIA_MAX_REPLANNING_TRIES")):
            kwargs["max_replanning_tries"] = int(_v)
        if (_v := os.environ.get("GAIA_TEST_IDX")):
            kwargs["test_idx"] = [int(x) for x in _v.split(",") if x.strip()]
        # Steady-concurrency load mode: GAIA_STEADY=1 holds max_workers tasks
        # in-flight (refilling a cycled seeded queue) for GAIA_STEADY_WALL_S secs.
        if os.environ.get("GAIA_STEADY", "").lower() in ("1", "true", "yes"):
            kwargs["steady"] = True
            kwargs["steady_wall_s"] = float(os.environ.get("GAIA_STEADY_WALL_S", "600"))
        # Open-loop Poisson-arrival load mode: GAIA_RPS=λ submits tasks at λ/s
        # (exponential inter-arrivals) for GAIA_RPS_WALL_S secs, then drains.
        # max_workers stays a cap; GAIA_RPS_SEED makes the arrival stream
        # reproducible. Takes precedence over GAIA_STEADY.
        if (_v := os.environ.get("GAIA_RPS")):
            kwargs["rps"] = float(_v)
            kwargs["rps_wall_s"] = float(os.environ.get("GAIA_RPS_WALL_S", "600"))
            kwargs["rps_seed"] = int(os.environ.get("GAIA_RPS_SEED", "0"))
            # default: submit the finite task set once (like sweep). Set
            # GAIA_RPS_CYCLE=1 to repeat it for sustained load-generation.
            kwargs["rps_cycle"] = os.environ.get(
                "GAIA_RPS_CYCLE", "").lower() in ("1", "true", "yes")

        evaluate_on_gaia(max_workers=max_workers, subset=subset, **kwargs)
    else:
        q = (
            " ".join(sys.argv[2:])
            if len(sys.argv) > 2
            else "According to wikipedia, when was The Battle of Diamond Rock?"
        )
        logger.success(process_single_task(q))
