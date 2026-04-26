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

import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from camel.models import ModelFactory
from camel.tasks import Task
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
_RESULTS_DIR = _SCRIPT_DIR / "results"
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


def make_model(role: str, **extra_config):
    """Build a vLLM-backed CAMEL model for the given role."""
    ep = _resolve_endpoint(role)
    model_type = _discover_model_name(ep.url, ep.api_key)
    cfg: Dict[str, Any] = {"temperature": 0.5, **ep.extra_config, **extra_config}
    logger.info(f"[vLLM] role={role:<18s} url={ep.url:<32s} model={model_type}")
    return ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=model_type,
        model_config_dict=cfg,
        url=ep.url,
        api_key=ep.api_key,
    )


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

    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ],
    )

    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ],
    )

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
    coordinator_agent_kwargs = {"model": make_model("coordinator")}
    task_agent_kwargs        = {"model": make_model("task")}
    answerer_agent_kwargs    = {"model": make_model("answerer")}

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


def _proc_init(data_dir: str, save_to: str) -> None:
    """Per-worker-process initializer. Builds the Workforce once and stashes
    it on a module global so every task this worker runs reuses it."""
    global _proc_workforce, _proc_benchmark, _proc_worker_id
    _proc_worker_id = os.getpid()
    _proc_benchmark = GAIABenchmark(data_dir=data_dir, save_to=save_to)
    # _proc_benchmark.load()
    _proc_workforce = construct_workforce(worker_id=_proc_worker_id)
    logger.info(f"[worker pid={_proc_worker_id}] initialized")


def _proc_run_one(
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

    success = False
    tries = 0
    trajectory_with_retry: List[dict] = []
    final_result: Optional[Dict[str, Any]] = None

    while not success and tries < max_tries:
        tries += 1
        logger.info(
            f"Attempt {tries}/{max_tries} for task {task['task_id']} "
            f"(worker pid={_proc_worker_id})"
        )
        try:
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
            success = score == True
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

    return final_result


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


def evaluate_on_gaia(level: int = 1, on: str = "valid", max_tries: int = 3,
                     test_idx: Optional[List[int]] = None,
                     save_result: bool = True,
                     max_workers: int = 1,
                     subset: Optional[int] = None):
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

    save_path = str(_RESULTS_DIR / f"workforce_{level}_pass{max_tries}_vllm.json")
    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR)

    benchmark = GAIABenchmark(data_dir=str(_DATA_DIR), save_to=save_path)

    if max_workers <= 1:
        workforce = construct_workforce(worker_id=0)
        result = benchmark.run_workforce_with_retry(
            workforce,
            on=on,
            level=level,
            idx=test_idx,
            subset=subset,
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=2,
        )
    else:
        result = _run_gaia_parallel(
            benchmark,
            on=on,
            level=level,
            test_idx=test_idx,
            max_tries=max_tries,
            max_replanning_tries=2,
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
        # Optional CLI: `python run_gaia_workforce_vllm_flex.py gaia [max_workers] [subset]`
        # Or via env: `GAIA_MAX_WORKERS=4 GAIA_SUBSET=20 python ...`
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            max_workers = int(sys.argv[2])
        else:
            max_workers = int(os.environ.get("GAIA_MAX_WORKERS", "1"))
        if len(sys.argv) > 3 and sys.argv[3].isdigit():
            subset = int(sys.argv[3])
        else:
            _env_subset = os.environ.get("GAIA_SUBSET")
            subset = int(_env_subset) if _env_subset else None
        evaluate_on_gaia(max_workers=max_workers, subset=subset)
    else:
        q = (
            " ".join(sys.argv[2:])
            if len(sys.argv) > 2
            else "According to wikipedia, when was The Battle of Diamond Rock?"
        )
        logger.success(process_single_task(q))
