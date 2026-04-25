"""
Flexible vLLM version of run_gaia_workforce.py.

All 12 model call-sites in the GAIA workforce are routed through `make_model(role)`
or `_resolve_endpoint(role)` (for tools that call vLLM directly):

  Workforce orchestrators (3): coordinator, task, answerer
  Worker agents (3):           web, document, reasoning
  Toolkit-internal models (6): image, audio, browser_web, browser_planning,
                               video (vLLM-served VL model — bypasses CAMEL
                                      and calls OpenAI-compatible API directly
                                      because CAMEL has no video message type),
                               and document (reused for DocumentProcessingToolkit's
                                             long-doc re-ranking inside
                                             _post_process_result)

Each role looks up its endpoint in VLLM_ENDPOINTS; missing roles fall back to
ROLE_FALLBACK, then to the "default" endpoint.

Two extreme configurations:

  (A) 6 agents share one vLLM server
      VLLM_ENDPOINTS = {"default": VllmEndpoint(url="http://localhost:8000/v1")}

  (B) each primary role gets its own vLLM server
      VLLM_ENDPOINTS = {
          "default":     VllmEndpoint(url="http://localhost:8000/v1"),
          "web":         VllmEndpoint(url="http://localhost:8001/v1"),
          "document":    VllmEndpoint(url="http://localhost:8002/v1"),
          "reasoning":   VllmEndpoint(url="http://localhost:8003/v1"),
          "coordinator": VllmEndpoint(url="http://localhost:8004/v1"),
          "task":        VllmEndpoint(url="http://localhost:8005/v1"),
          "answerer":    VllmEndpoint(url="http://localhost:8006/v1"),
      }

  Anything in between works too. The 4 sub-agent roles inside worker toolkits
  (`image`, `audio`, `browser_web`, `browser_planning`) piggyback on their
  parent worker unless explicitly overridden.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    model_type: Optional[str] = None           # None → auto-discover via /v1/models
    api_key: str = "EMPTY"
    extra_config: Dict[str, Any] = field(default_factory=dict)


# Default endpoint URL; override by `VLLM_URL=... python run_gaia_workforce_vllm_flex.py`
_DEFAULT_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")

# ─── EDIT THIS DICT ──────────────────────────────────────────────────────────
# "default" must exist. Add per-role entries to split across multiple servers.
VLLM_ENDPOINTS: Dict[str, VllmEndpoint] = {
    "default": VllmEndpoint(url=_DEFAULT_URL),

    # Primary roles (6 agents the Workforce actually runs):
    # "web":         VllmEndpoint(url="http://localhost:8001/v1"),
    # "document":    VllmEndpoint(url="http://localhost:8002/v1"),
    # "reasoning":   VllmEndpoint(url="http://localhost:8003/v1"),
    # "coordinator": VllmEndpoint(url="http://localhost:8004/v1"),
    # "task":        VllmEndpoint(url="http://localhost:8005/v1"),
    # "answerer":    VllmEndpoint(url="http://localhost:8006/v1"),

    # Sub-agents inside worker toolkits. If omitted they follow ROLE_FALLBACK:
    # "image":            VllmEndpoint(url="..."),
    # "audio":            VllmEndpoint(url="..."),
    # "browser_web":      VllmEndpoint(url="..."),
    # "browser_planning": VllmEndpoint(url="..."),
}

# When a role isn't explicitly in VLLM_ENDPOINTS, try its fallback first.
ROLE_FALLBACK: Dict[str, str] = {
    "image":            "document",
    "audio":            "document",
    "browser_web":      "web",
    "browser_planning": "web",
    "video":            "image",   # video shares VL endpoint with image by default
}


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
    fallback = ROLE_FALLBACK.get(role)
    if fallback and fallback in VLLM_ENDPOINTS:
        return VLLM_ENDPOINTS[fallback]
    return VLLM_ENDPOINTS["default"]


def make_model(role: str, **extra_config):
    """Build a vLLM-backed CAMEL model for the given role."""
    ep = _resolve_endpoint(role)
    model_type = ep.model_type or _discover_model_name(ep.url, ep.api_key)
    cfg: Dict[str, Any] = {"temperature": 0, **ep.extra_config, **extra_config}
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

def construct_agent_list() -> List[Dict[str, Any]]:
    web_model                = make_model("web")
    document_processing_model = make_model("document")
    reasoning_model          = make_model("reasoning")
    image_analysis_model     = make_model("image")
    audio_reasoning_model    = make_model("audio")
    browser_web_model        = make_model("browser_web")
    browser_planning_model   = make_model("browser_planning")

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(
        cache_dir="tmp",
        text_processing_model=make_model("document"),
    )
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)

    # Video toolkit: bypass CAMEL, talk to vLLM's OpenAI-compatible API directly
    # because CAMEL has no native video message type.
    video_ep = _resolve_endpoint("video")
    video_model_name = video_ep.model_type or _discover_model_name(video_ep.url, video_ep.api_key)
    video_analysis_toolkit = VideoAnalysisToolkit(
        download_directory="tmp/video",
        vllm_url=video_ep.url,
        vllm_api_key=video_ep.api_key,
        vllm_model=video_model_name,
    )
    audio_analysis_toolkit = AudioAnalysisToolkit(
        cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model,
    )
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    browser_simulator_toolkit = AsyncBrowserToolkit(
        headless=True,
        cache_dir="tmp/browser",
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


def construct_workforce() -> OwlGaiaWorkforce:
    coordinator_agent_kwargs = {"model": make_model("coordinator")}
    task_agent_kwargs        = {"model": make_model("task")}
    answerer_agent_kwargs    = {"model": make_model("answerer")}

    workforce = OwlGaiaWorkforce(
        "Gaia Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
        answerer_agent_kwargs=answerer_agent_kwargs,
    )

    for agent_dict in construct_agent_list():
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


def evaluate_on_gaia(level: int = 1, on: str = "valid", max_tries: int = 3,
                     test_idx: Optional[List[int]] = None,
                     save_result: bool = True):
    """Full GAIA benchmark sweep."""
    if test_idx is None:
        test_idx = [1]

    save_path = f"results/workforce/workforce_{level}_pass{max_tries}_vllm.json"
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")

    benchmark = GAIABenchmark(data_dir="data/gaia", save_to=save_path)
    workforce = construct_workforce()

    result = benchmark.run_workforce_with_retry(
        workforce,
        on=on,
        level=level,
        idx=test_idx,
        save_result=save_result,
        max_tries=max_tries,
        max_replanning_tries=2,
    )
    logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "gaia":
        evaluate_on_gaia()
    else:
        q = (
            " ".join(sys.argv[2:])
            if len(sys.argv) > 2
            else "According to wikipedia, when was The Battle of Diamond Rock?"
        )
        logger.success(process_single_task(q))
