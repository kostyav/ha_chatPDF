"""Parser service — consumes parse.requests, produces parse.results.

Dependency note: this container pins transformers>=4.48 so that Docling's
rt_detr_v2 layout model loads correctly.  That version is incompatible with
the colpali-engine used by the visual_indexer, which is why these two services
must live in separate containers.
"""
import json
import logging
import os
import sys
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer

sys.path.insert(0, "/app")
import shared.schemas as schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [parser] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
PARSED_DIR      = Path(os.environ.get("PARSED_DIR", "/data/parsed"))
DPI             = int(os.environ.get("DPI", "300"))


def _parse_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> dict:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

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

    return {
        "markdown": doc.export_to_markdown(),
        "tables_md": [t.export_to_markdown() for t in doc.tables],
    }


def main() -> None:
    consumer = KafkaConsumer(
        schemas.TOPIC_PARSE_REQUESTS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="parser",
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    log.info("Parser service ready — waiting for parse requests …")

    for msg in consumer:
        req = msg.value
        pdf_path = Path(req["pdf_path"])
        pdf_id   = req["pdf_id"]
        log.info("Parsing %s", pdf_id)
        try:
            parsed = _parse_pdf(pdf_path, PARSED_DIR / pdf_id, DPI)
            producer.send(
                schemas.TOPIC_PARSE_RESULTS,
                schemas.parse_result(pdf_id, parsed["markdown"], parsed["tables_md"]),
            )
            producer.flush()
            log.info("Published parse.results for %s (%d chars)", pdf_id, len(parsed["markdown"]))
        except Exception:
            log.exception("Failed to parse %s", pdf_id)


if __name__ == "__main__":
    main()
