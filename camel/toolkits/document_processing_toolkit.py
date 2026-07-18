from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits import AudioAnalysisToolkit, ExcelToolkit, ImageAnalysisToolkit
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, ModelPlatformType
from camel.agents import ChatAgent
from docx2markdown._docx_to_markdown import docx_to_markdown
import requests
import mimetypes
import json
from retry import retry
from typing import Any, List, Optional, Tuple
from loguru import logger
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse
import os
import subprocess
import contextvars
import hashlib
import xmltodict
import nest_asyncio
nest_asyncio.apply()


class DocumentProcessingToolkit(BaseToolkit):
    r"""A class representing a toolkit for processing document and return the content of the document.

    This class provides method for processing docx, pdf, pptx, etc. It cannot process excel files.
    """
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        image_analysis_model: Optional[BaseModelBackend] = None,
        text_processing_model: Optional[BaseModelBackend] = None,
        enable_model_features: bool = True,
    ):
        self.image_tool = (
            ImageAnalysisToolkit(model=image_analysis_model)
            if enable_model_features
            else None
        )
        self.audio_tool = AudioAnalysisToolkit() if enable_model_features else None
        self.excel_tool = ExcelToolkit()
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }
        self.text_processing_model = text_processing_model

        self.cache_dir = "tmp/"
        if cache_dir:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if enable_model_features and self.text_processing_model is None:
            self.text_processing_model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.O3_MINI,
                model_config_dict={"temperature": 0.0}
            )
    
    def _resolve_document_parser(self, document_path: str) -> str:
        """Resolve the parser once so replay receives an explicit choice."""

        parsed = urlparse(document_path)
        path = parsed.path if parsed.scheme in {"http", "https"} else document_path
        suffix = os.path.splitext(path)[1].lower()
        parsers = {
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".mp3": "audio",
            ".wav": "audio",
            ".txt": "text",
            ".xls": "excel",
            ".xlsx": "excel",
            ".csv": "excel",
            ".zip": "zip",
            ".json": "json",
            ".jsonl": "json",
            ".jsonld": "json",
            ".py": "python",
            ".xml": "xml",
            ".docx": "docx",
            ".pptx": "pptx",
            ".pdf": "pdf",
        }
        if suffix in parsers:
            return parsers[suffix]
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return "webpage" if self._is_webpage(document_path) else "generic"
        return "generic"

    def extract_document_raw(
        self, document_path: str, parser: str
    ) -> Tuple[bool, Any]:
        """Extract content without invoking any language or vision model.

        ``parser`` is explicit rather than inferred inside the replay tool so
        the recorded invocation fully describes the executed operation.
        """

        parsed_url = urlparse(document_path)
        is_url = parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)

        if parser in {"image", "audio"}:
            return False, f"{parser} content requires a model capability"
        if parser in {"text", "python"}:
            path = self._download_file(document_path) if is_url else document_path
            with open(path, "r", encoding="utf-8") as handle:
                return True, handle.read()
        if parser == "excel":
            path = self._download_file(document_path) if is_url else document_path
            return True, self.excel_tool.extract_excel_content(path)
        if parser == "zip":
            path = self._download_file(document_path) if is_url else document_path
            return True, f"The extracted files are: {self._unzip_file(path)}"
        if parser == "json":
            path = self._download_file(document_path) if is_url else document_path
            with open(path, "r", encoding="utf-8") as handle:
                if path.endswith(".jsonl"):
                    return True, handle.read()
                return True, json.load(handle)
        if parser == "xml":
            path = self._download_file(document_path) if is_url else document_path
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            try:
                return True, xmltodict.parse(content)
            except Exception:
                return True, content
        if parser == "webpage":
            return True, self._extract_webpage_content(document_path)
        if parser == "docx":
            path = self._download_file(document_path) if is_url else document_path
            output_path = os.path.join(
                self.cache_dir, f"{os.path.basename(path)}.md"
            )
            docx_to_markdown(path, output_path)
            with open(output_path, "r", encoding="utf-8") as handle:
                return True, handle.read()
        if parser == "pptx":
            from unstructured.partition.auto import partition

            path = self._download_file(document_path) if is_url else document_path
            return True, [item.text for item in partition(path)]
        if parser == "pdf":
            from PyPDF2 import PdfReader

            path = self._download_file(document_path) if is_url else document_path
            with open(path, "rb") as handle:
                reader = PdfReader(handle)
                return True, "".join(page.extract_text() or "" for page in reader.pages)
        if parser == "generic":
            from unstructured.partition.auto import partition

            path = self._download_file(document_path) if is_url else document_path
            return True, [item.text for item in partition(path)]
        return False, f"Unsupported document parser: {parser}"

    # This remains the Agent-facing orchestration API. Replay capture suppresses
    # the outer call and records ``document_extract_raw``, model requests, and
    # ``document_select_chunks`` independently.
    def extract_document_content(self, document_path: str, query: str = None) -> Tuple[bool, Any]:
        r"""Extract the content of a given document (or url) and return the processed text.
        It may filter out some information, resulting in inaccurate content.

        Args:
            document_path (str): The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc.
            query (str): The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the document was processed successfully, and the content of the document (if success).
        """
        logger.debug(
            "Calling extract_document_content function with "
            f"document_path=`{document_path}`"
        )
        parser = self._resolve_document_parser(document_path)
        if parser == "image":
            if self.image_tool is None:
                return False, "Image model capability is disabled"
            return True, self.image_tool.ask_question_about_image(
                document_path, "Please make a detailed caption about the image."
            )
        if parser == "audio":
            if self.audio_tool is None:
                return False, "Audio model capability is disabled"
            return True, self.audio_tool.ask_question_about_audio(
                document_path, "Please transcribe the audio content to text."
            )
        from camel.utils.replay_capture import run_tool_primitive

        try:
            success, content = run_tool_primitive(
                name="document_extract_raw",
                arguments={"document_path": document_path, "parser": parser},
                function=lambda: self.extract_document_raw(document_path, parser),
                toolkit="DocumentExtractionPrimitive",
            )
        except Exception as exc:
            logger.error(f"Error occurred while processing document: {exc}")
            return False, f"Error occurred while processing document: {exc}"
        if not success:
            return success, content
        if parser in {"text", "webpage", "pdf"} and isinstance(content, str):
            content = self._post_process_result(content, query)
        return True, content
    
    
    def _post_process_result(self, result: str, query: str) -> str:
        r"""Identify whether the result is too long. If so, split it into multiple parts, and leverage a model to identify which part contains the relevant information.
        """
        import concurrent.futures
        
        def _identify_relevant_part(part_idx: int, part: str, query: str, _process_model: BaseModelBackend = None) -> Tuple[bool, str]:
            agent = ChatAgent(
                model=_process_model
            )
            
            prompt = f"""
I have retrieved some information from a long document. 
Now I have split the document into multiple parts. Your task is to identify whether the given part contains the relevant information based on the query.

If it does, return only "True". If it doesn't, return only "False". Do not return any other information.

Document part:
<document_part>
{part}
</document_part>

Query:
<query>
{query}
</query>
"""
            
            response = agent.step(prompt)
            if "true" in response.msgs[0].content.lower():
                return True, part_idx, part
            else:
                return False, part_idx, part
                
            
        max_length = 200000
        split_length = 40000
        
        if len(result) > max_length:
            # split the result into multiple parts
            logger.debug(f"The original result is too long. Splitting it into multiple parts. query: {query}")
            parts = [result[i:i+split_length] for i in range(0, len(result), split_length)]
            result_cache = {}
            # use concurrent.futures to process the parts
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(
                        contextvars.copy_context().run,
                        _identify_relevant_part,
                        part_idx,
                        part,
                        query,
                        self.text_processing_model,
                    )
                    for part_idx, part in enumerate(parts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    is_relevant, part_idx, part = future.result()
                    if is_relevant:
                        result_cache[part_idx] = part
            selected_indices = sorted(result_cache)
            from camel.utils.replay_capture import run_tool_primitive

            return run_tool_primitive(
                name="document_select_chunks",
                arguments={
                    "selected_indices": selected_indices,
                    "split_length": split_length,
                    "max_length": max_length,
                    "source_sha256": hashlib.sha256(result.encode()).hexdigest(),
                },
                function=lambda: self.select_document_chunks(
                    result,
                    selected_indices,
                    split_length=split_length,
                    max_length=max_length,
                ),
                toolkit="DocumentExtractionPrimitive",
            )
        
        else:
            return result

    def select_document_chunks(
        self,
        result: str,
        selected_indices: List[int],
        *,
        split_length: int = 40000,
        max_length: int = 200000,
    ) -> str:
        """Deterministically assemble chunk indexes selected by an LM event."""

        parts = [result[i : i + split_length] for i in range(0, len(result), split_length)]
        invalid = [index for index in selected_indices if index < 0 or index >= len(parts)]
        if invalid:
            raise ValueError(f"Invalid document chunk indexes: {invalid}")
        filtered = "".join(f"{parts[index]}..." for index in selected_indices)
        filtered += (
            "(The above is the re-assembled result of the document, because "
            "the original document is too long. If empty, it means no relevant "
            "information found.)"
        )
        if len(filtered) > max_length:
            filtered = filtered[:max_length]
        logger.debug(f"split context length: {len(filtered)}")
        return filtered


    def _is_webpage(self, url: str) -> bool:
        r"""Judge whether the given URL is a webpage."""
        try:
            parsed_url = urlparse(url)
            is_url = all([parsed_url.scheme, parsed_url.netloc])
            if not is_url:
                return False

            path = parsed_url.path
            file_type, _ = mimetypes.guess_type(path)
            if 'text/html' in file_type:
                return True
            
            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get("Content-Type", "").lower()
            
            if "text/html" in content_type:
                return True
            else:
                return False
        
        except requests.exceptions.RequestException as e:
            # raise RuntimeError(f"Error while checking the URL: {e}")
            logger.warning(f"Error while checking the URL: {e}")
            return False

        except TypeError:
            return True
    

    # Bounded retry + explicit (connect, read) timeout. Without a timeout a
    # webpage GET can hang forever (server accepts the connection then never
    # responds), and the default `retry` tries=-1 made that an *infinite* retry
    # loop — a single slow URL wedged the whole GAIA task. Cap both.
    @retry(requests.RequestException, tries=3, delay=2, backoff=2, max_delay=30)
    def _extract_webpage_content_with_html2text(self, url: str) -> str:
        import html2text
        h = html2text.HTML2Text()
        response = requests.get(url, headers=self.headers, timeout=(5, 15))
        html_content = response.text
        
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        extracted_text = h.handle(html_content)
        return extracted_text
    
    @retry(requests.RequestException, tries=3, delay=2, backoff=2, max_delay=30)
    def _extract_webpage_content_with_beautifulsoup(self, url: str) -> str:
        response = requests.get(url, headers=self.headers, timeout=(5, 15))
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        extracted_text = soup.get_text()
        return extracted_text

    def _looks_like_challenge(self, text: str) -> bool:
        """Heuristic: did a bare-HTTP fetch get a bot-protection / JS-challenge
        page (or near-empty body) instead of the real content?"""
        if not text or len(text.strip()) < 200:
            return True
        low = text.lower()
        markers = (
            "just a moment", "checking your browser", "verify you are human",
            "security verification", "enable javascript", "captcha",
            "access denied", "cloudflare", "ddos protection",
        )
        return any(m in low for m in markers)

    async def _browser_fetch_html(self, url: str) -> str:
        """Render the page in a real headless Chromium and return its DOM HTML.
        Runs JS and carries a browser fingerprint, so it gets through most
        JS-gated / bot-protected pages that the bare requests.get path can't."""
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                ctx = await browser.new_context(user_agent=self.headers["User-Agent"])
                page = await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    # let JS / a bot challenge settle (bounded)
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                return await page.content()
            finally:
                await browser.close()

    def _extract_webpage_content_with_browser(self, url: str) -> str:
        import html2text
        # nest_asyncio (applied at import) lets asyncio.run work even when the
        # caller is already inside an event loop.
        html_content = asyncio.run(self._browser_fetch_html(url))
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        return h.handle(html_content)


    @retry(RuntimeError, tries=3, delay=5, backoff=2, max_delay=30)
    def _extract_webpage_content(self, url: str) -> str:
        api_key = os.getenv("FIRECRAWL_API_KEY")

        # Skip Firecrawl entirely without an API key — its constructor raises
        # ValueError before reaching the try block, so the exception cannot be
        # caught here. Use the local html2text path, and fall back to a real
        # headless browser when bare HTTP fails or gets a bot-challenge page.
        if not api_key:
            try:
                text = self._extract_webpage_content_with_html2text(url)
                if self._looks_like_challenge(text):
                    logger.warning(
                        f"html2text got a challenge/empty page for {url}; "
                        f"retrying via headless browser")
                    return self._extract_webpage_content_with_browser(url)
                return text
            except Exception as e:
                logger.warning(
                    f"html2text failed for {url} ({type(e).__name__}: {e}); "
                    f"falling back to headless browser")
                return self._extract_webpage_content_with_browser(url)

        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=api_key)
            data = app.crawl_url(
                url,
                params={
                    'limit': 1,
                    'scrapeOptions': {'formats': ['markdown']},
                },
            )
        except Exception as e:
            if "429" in str(e):
                # too many requests — keep original retry behavior
                logger.error(f"Error: {e}")
                raise RuntimeError(f"Error: {e}")
            # any other failure (bad key / 401/402/403 / SDK API mismatch /
            # network) falls back to local html2text
            logger.warning(
                f"Firecrawl failed ({type(e).__name__}: {e}); "
                f"falling back to html2text."
            )
            return self._extract_webpage_content_with_html2text(url)

        logger.debug(f"Extracted data from {url} using firecrawl: {data}")
        if len(data['data']) == 0:
            if data['success']:
                logger.debug("Trying to use html2text to get the text.")
                # try using html2text to get the text
                extracted_text = self._extract_webpage_content_with_html2text(url)
                logger.debug(f"The extracted text from html2text is: {extracted_text}")
                
                if len(extracted_text) == 0:
                    return "No content found on the webpage."
                else:
                    return extracted_text

            else:
                return "Error while crawling the webpage."

        return str(data['data'][0]['markdown'])
    

    def _download_file(self, url: str):
        r"""Download a file from a URL and save it to the cache directory."""
        try:
            response = requests.get(url, stream=True, headers=self.headers, timeout=(5, 15))
            response.raise_for_status()
            file_name = url.split("/")[-1]  

            file_path = os.path.join(self.cache_dir, file_name)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")


    def _get_formatted_time(self) -> str:
        import time
        return time.strftime("%m%d%H%M")

    
    def _unzip_file(self, zip_path: str) -> List[str]:
        if not zip_path.endswith('.zip'):
            raise ValueError("Only .zip files are supported")
        
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_path = os.path.join(self.cache_dir, zip_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to unzip file: {e}")

        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                extracted_files.append(os.path.join(root, file))
        
        return extracted_files


    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.extract_document_content),
        ]
