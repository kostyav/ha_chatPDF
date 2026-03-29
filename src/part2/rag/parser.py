"""PDF parsing with Docling — extracts text, Markdown tables, and page images."""
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class ParsedDoc:
    pdf_id: str
    markdown: str
    page_images: list[Path] = field(default_factory=list)
    tables_md: list[str] = field(default_factory=list)


def parse_pdf(pdf_path: Path, output_dir: Path, dpi: int = 300) -> ParsedDoc:
    """Parse a PDF with Docling. Returns Markdown text, table Markdown, and page image paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = PdfPipelineOptions()
    opts.images_scale = dpi / 72
    opts.generate_page_images = True
    opts.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Save per-page images
    images: list[Path] = []
    for page_no, page in doc.pages.items():
        img_path = output_dir / f"page_{page_no}.png"
        if hasattr(page, "image") and page.image is not None:
            page.image.pil_image.save(img_path)
            images.append(img_path)

    return ParsedDoc(
        pdf_id=pdf_path.stem,
        markdown=doc.export_to_markdown(),
        page_images=images,
        tables_md=[t.export_to_markdown() for t in doc.tables],
    )
