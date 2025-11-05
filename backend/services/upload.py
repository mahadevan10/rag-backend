"""File upload processing and ingestion pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from pypdf import PdfReader

from langchain_core.documents import Document

from ..config import Settings
from .document_store import DocumentStore

logger = logging.getLogger(__name__)


class UploadService:
    """Process uploaded PDF files and ingest them into the document store."""

    def __init__(self, doc_store: DocumentStore, settings: Settings) -> None:
        self._doc_store = doc_store
        self._settings = settings
        self._jobs: Dict[str, Dict] = {}
        self._ocr_results: Dict[str, List[Dict]] = {}  # Store OCR output for testing
    
    def _create_smart_chunks(
        self,
        page_texts: List[tuple],
        filename: str,
        doc_id: str,
        total_pages: int,
        job_id: str,
        **kwargs  # Accept extra args for compatibility
    ) -> List[Document]:
        """Create one chunk per page (no further splitting)."""
        all_chunks = []
        for page_num, page_text in page_texts:
            if not page_text.strip():
                continue
            all_chunks.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_id": page_num,
                        "doc_id": doc_id,
                        "total_pages": total_pages,
                        "upload_job_id": job_id,
                    },
                )
            )
        logger.info(
            "Created %d page-based chunks for %s",
            len(all_chunks),
            filename,
        )
        return all_chunks

    def create_job(self, filenames: List[str]) -> str:
        job_id = str(uuid4())
        self._jobs[job_id] = {
            "status": "processing",
            "filenames": filenames,
            "progress": 0,
        }
        return job_id

    def process_upload(self, job_id: str, files_data: List[Dict]) -> None:
        try:
            for file_data in files_data:
                filename = file_data["filename"]
                content = file_data["content"]
                save_path = Path("documents") / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(content)

                reader = PdfReader(str(save_path))
                docs: List[Document] = []
                doc_id = str(uuid4())
                total_pages = len(reader.pages)

                use_ocr = bool(self._settings.enable_ocr)
                poppler_path = self._settings.poppler_path
                tesseract_cmd = self._settings.tesseract_cmd

                if use_ocr and tesseract_cmd:
                    try:
                        import pytesseract  # type: ignore

                        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                    except Exception as exc:  # pragma: no cover - optional dependency
                        logger.warning("Failed to configure pytesseract: %s", exc)
                        use_ocr = False

                # Extract all pages first
                page_texts = []
                ocr_log = []  # Track OCR usage for testing
                
                for index, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    used_ocr = False
                    ocr_confidence = None

                    if not text.strip() and use_ocr:
                        try:
                            from pdf2image import convert_from_path  # type: ignore
                            import pytesseract  # type: ignore

                            logger.info(f"📸 Running OCR on {filename} page {index + 1} (no text found)")
                            
                            images = convert_from_path(
                                str(save_path),
                                first_page=index + 1,
                                last_page=index + 1,
                                poppler_path=poppler_path,
                            )
                            if images:
                                # Get OCR data with confidence
                                ocr_data = pytesseract.image_to_data(images[0], output_type=pytesseract.Output.DICT)
                                text = pytesseract.image_to_string(images[0]) or ""
                                
                                # Calculate average confidence (excluding -1 values)
                                confidences = [int(c) for c in ocr_data['conf'] if int(c) != -1]
                                ocr_confidence = sum(confidences) / len(confidences) if confidences else 0
                                
                                used_ocr = True
                                
                                logger.info(
                                    f"✅ OCR completed for page {index + 1}: "
                                    f"{len(text)} chars extracted, "
                                    f"avg confidence: {ocr_confidence:.1f}%"
                                )
                                
                                # Store detailed OCR info
                                ocr_log.append({
                                    "page": index + 1,
                                    "chars_extracted": len(text),
                                    "confidence": round(ocr_confidence, 2),
                                    "text_preview": text[:200] if text else "(empty)",
                                    "word_count": len(text.split()) if text else 0,
                                })
                        except Exception as exc:  # pragma: no cover - best effort OCR
                            logger.error(f"❌ OCR failed for {filename} page {index + 1}: {exc}")
                            ocr_log.append({
                                "page": index + 1,
                                "error": str(exc),
                                "chars_extracted": 0,
                            })

                    page_texts.append((index + 1, text))
                    
                    # Log extraction method for each page
                    if used_ocr:
                        logger.debug(f"Page {index + 1}: OCR (confidence: {ocr_confidence:.1f}%)")
                    elif text.strip():
                        logger.debug(f"Page {index + 1}: Native PDF text ({len(text)} chars)")
                    else:
                        logger.debug(f"Page {index + 1}: Empty (no text or OCR disabled)")
                
                # Store OCR results for this job
                if ocr_log:
                    self._ocr_results[job_id] = ocr_log
                    logger.info(f"📊 OCR Summary for {filename}: {len(ocr_log)} pages processed with OCR")
                
                # Now create smart chunks with semantic boundaries and overlap
                docs = self._create_smart_chunks(
                    page_texts=page_texts,
                    filename=filename,
                    doc_id=doc_id,
                    total_pages=total_pages,
                    job_id=job_id
                )

                # Print metadata for all chunks for testing
                for i, chunk in enumerate(docs):
                    logger.info(f"Chunk {i+1} metadata: {chunk.metadata}")

                if docs:
                    self._doc_store.add_documents(docs)

            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["progress"] = 100
        except Exception as exc:  # pragma: no cover - fail safe
            logger.exception("Upload processing failed")
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = str(exc)

    def get_job_status(self, job_id: str) -> Dict:
        status = self._jobs.get(job_id, {"status": "not_found"})
        
        # Include OCR results if available
        if job_id in self._ocr_results:
            status["ocr_results"] = self._ocr_results[job_id]
        
        return status
    
    def get_ocr_results(self, job_id: str) -> Dict:
        """Get detailed OCR results for testing/debugging.
        
        Returns:
            Dictionary with OCR statistics and per-page results
        """
        if job_id not in self._ocr_results:
            return {
                "job_id": job_id,
                "ocr_used": False,
                "message": "No OCR data available for this job"
            }
        
        ocr_data = self._ocr_results[job_id]
        
        # Calculate summary statistics
        total_pages = len(ocr_data)
        successful_pages = sum(1 for page in ocr_data if "error" not in page)
        failed_pages = total_pages - successful_pages
        total_chars = sum(page.get("chars_extracted", 0) for page in ocr_data)
        avg_confidence = sum(page.get("confidence", 0) for page in ocr_data if "confidence" in page) / successful_pages if successful_pages > 0 else 0
        
        return {
            "job_id": job_id,
            "ocr_used": True,
            "summary": {
                "total_pages_processed": total_pages,
                "successful": successful_pages,
                "failed": failed_pages,
                "total_characters_extracted": total_chars,
                "average_confidence": round(avg_confidence, 2),
            },
            "pages": ocr_data,
        }
