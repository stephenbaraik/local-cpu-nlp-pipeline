from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from pipeline.stages import DocContext, register


@dataclass
class ExtractStage:
    name: str = "extract"
    version: str = "1"
    depends_on: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ("MAX_PAGES",)

    def run(self, doc: DocContext) -> dict:
        pdf = pymupdf.open("pdf", doc.pdf_bytes)
        try:
            pages_total = pdf.page_count
            max_pages = doc.config.max_pages
            pages_read = pages_total if max_pages <= 0 else min(max_pages, pages_total)
            pages = [pdf[i].get_text() for i in range(pages_read)]
        finally:
            pdf.close()
        return {
            "pages_total": pages_total,
            "pages_read": pages_read,
            "pages": pages,
        }


register(ExtractStage())
