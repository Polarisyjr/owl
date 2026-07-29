from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits import AudioAnalysisToolkit, ExcelToolkit, ImageAnalysisToolkit
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, ModelPlatformType
from camel.agents import ChatAgent
from docx2markdown._docx_to_markdown import docx_to_markdown
import requests
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
            return self._probe_remote_parser(document_path)
        return "generic"

    @staticmethod
    def _parser_from_content_type(content_type: str) -> Optional[str]:
        """Map an HTTP media type to one of the explicit document parsers."""

        media_type = content_type.partition(";")[0].strip().lower()
        if not media_type:
            return None
        if media_type == "application/pdf":
            return "pdf"
        if media_type in {"text/html", "application/xhtml+xml"}:
            return "webpage"
        if media_type.startswith("image/"):
            return "image"
        if media_type.startswith("audio/"):
            return "audio"
        if media_type in {"text/plain", "text/x-python"}:
            return "text"
        if media_type in {
            "application/json",
            "application/ld+json",
            "application/x-ndjson",
        }:
            return "json"
        if media_type in {"application/xml", "text/xml"}:
            return "xml"
        if media_type in {
            "application/zip",
            "application/x-zip-compressed",
        }:
            return "zip"
        if media_type in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
        }:
            return "excel"
        if media_type == (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ):
            return "docx"
        if media_type == (
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        ):
            return "pptx"
        return None

    def _parser_from_response_metadata(self, response: requests.Response) -> Optional[str]:
        """Infer a parser from a response's final URL and Content-Type."""

        final_path = urlparse(response.url).path
        suffix = os.path.splitext(final_path)[1].lower()
        suffix_parsers = {
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
        if suffix in suffix_parsers:
            return suffix_parsers[suffix]
        return self._parser_from_content_type(
            response.headers.get("Content-Type", "")
        )

    def _probe_remote_parser(self, url: str) -> str:
        """Resolve extensionless remote documents without decoding bytes as HTML.

        A HEAD response is useful but not authoritative: CDNs and bot-protection
        pages can report ``text/html`` for a URL whose GET body is a PDF.  For an
        ambiguous or HTML-looking HEAD response, inspect a small prefix of the
        real GET response as well.  PDF magic wins over response metadata.
        """

        head_parser = None
        head_error = None
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                headers=self.headers,
                timeout=(5, 10),
            )
            detected_parser = self._parser_from_response_metadata(response)
            if (
                response.status_code in {401, 403, 429}
                and detected_parser == "webpage"
            ):
                return "webpage"
            response.raise_for_status()
            head_parser = detected_parser
            # A non-HTML media type or a recognized suffix is sufficiently
            # specific. HTML gets confirmed with GET because it may be a
            # transient challenge page.
            if head_parser is not None and head_parser != "webpage":
                return head_parser
        except requests.RequestException as exc:
            head_error = exc

        probe_headers = dict(self.headers)
        probe_headers["Range"] = "bytes=0-1023"
        try:
            with requests.get(
                url,
                allow_redirects=True,
                headers=probe_headers,
                stream=True,
                timeout=(5, 15),
            ) as response:
                response_parser = self._parser_from_response_metadata(response)
                if (
                    response.status_code in {401, 403, 429}
                    and response_parser == "webpage"
                ):
                    return "webpage"
                response.raise_for_status()
                prefix = b""
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        prefix += chunk
                    if len(prefix) >= 1024:
                        break

                # ISO 32000 permits the PDF header within the first 1024 bytes.
                if b"%PDF-" in prefix:
                    return "pdf"

                if response_parser is not None:
                    return response_parser

                normalized = prefix.lstrip().lower()
                if normalized.startswith(
                    (b"<!doctype html", b"<html", b"<?xml")
                ):
                    return "webpage"
        except requests.RequestException as exc:
            if head_error is not None:
                logger.warning(
                    f"Could not determine remote document type for {url}: "
                    f"HEAD failed with {head_error}; GET probe failed with {exc}"
                )
            else:
                logger.warning(
                    f"Could not confirm remote document type for {url}: {exc}"
                )

        return head_parser or "generic"

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
        parsed_url = urlparse(url)
        if not (parsed_url.scheme in {"http", "https"} and parsed_url.netloc):
            return False
        return self._probe_remote_parser(url) == "webpage"
    

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
            except Exception as e:
                logger.warning(
                    f"html2text failed for {url} ({type(e).__name__}: {e}); "
                    f"falling back to headless browser")
            else:
                if not self._looks_like_challenge(text):
                    return text
                logger.warning(
                    f"html2text got a challenge/empty page for {url}; "
                    f"retrying via headless browser"
                )

            browser_text = self._extract_webpage_content_with_browser(url)
            if self._looks_like_challenge(browser_text):
                raise requests.RequestException(
                    "Bot-protection challenge remained after browser "
                    f"fallback for {url}"
                )
            return browser_text

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

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download {url}: {exc}"
            ) from exc


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
