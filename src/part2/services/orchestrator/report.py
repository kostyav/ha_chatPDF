"""Generate a PDF report from retrieval_log.jsonl.

Each entry becomes a section showing:
  - the query
  - retrieved text chunks with scores
  - retrieved page images (if any)

Usage:
  docker compose run --rm orchestrator python report.py
  docker compose run --rm orchestrator python report.py --log /data/logs/retrieval_log.jsonl \
                                                        --out /data/logs/report.pdf
"""
import argparse
import json
import os
from pathlib import Path

from fpdf import FPDF, XPos, YPos

RETRIEVAL_LOG = os.environ.get("RETRIEVAL_LOG", "/data/logs/retrieval_log.jsonl")
DEFAULT_OUT   = str(Path(RETRIEVAL_LOG).parent / "report.pdf")

ACCENT   = (52, 101, 164)   # section header blue
CHUNK_BG = (245, 245, 245)  # light grey chunk background

_FONT_DIR  = Path("/usr/share/fonts/truetype/dejavu")
_FONT_REG  = str(_FONT_DIR / "DejaVuSans.ttf")
_FONT_BOLD = str(_FONT_DIR / "DejaVuSans-Bold.ttf")


class ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font("DejaVu", style="",  fname=_FONT_REG)
        self.add_font("DejaVu", style="B", fname=_FONT_BOLD)
        self.add_font("DejaVu", style="I", fname=_FONT_REG)  # no oblique variant in package

    def header(self):
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, "RAG Retrieval Report", align="R")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, text: str) -> None:
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 7, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def meta_line(self, label: str, value: str) -> None:
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(28, 6, label)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(self.epw - 28, 6, value)

    def chunk_block(self, score: float, pdf_id: str, text: str) -> None:
        self.set_fill_color(*CHUNK_BG)
        x, y = self.get_x(), self.get_y()
        self.rect(x, y, self.epw, 5, style="F")   # coloured header strip
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(*ACCENT)
        self.cell(0, 5, f"  score {score:.3f}   pdf {pdf_id}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.set_font("DejaVu", "", 9)
        self.set_fill_color(*CHUNK_BG)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(2)

    def embed_image(self, path: str, caption: str) -> None:
        if not Path(path).exists():
            return
        max_w = self.epw
        max_h = 120
        if self.get_y() + max_h + 10 > self.h - self.b_margin:
            self.add_page()
        self.image(path, x=self.l_margin, w=max_w, h=max_h, keep_aspect_ratio=True)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(120, 120, 120)
        self.multi_cell(0, 5, caption)
        self.set_text_color(0, 0, 0)
        self.ln(3)


def build_report(log_path: Path, out_path: Path) -> None:
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not entries:
        print("Log file is empty — nothing to render.")
        return

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    for i, entry in enumerate(entries, start=1):
        pdf.add_page()

        # ── Query header ───────────────────────────────────────────────────────
        pdf.section_title(f"Query {i}")
        pdf.meta_line("Question:", entry.get("query", ""))
        pdf.meta_line("Timestamp:", entry.get("timestamp", ""))
        pdf.meta_line("Best score:", f"{entry.get('best_score', 0):.3f}")
        pdf.ln(4)

        # ── LLM Answer ────────────────────────────────────────────────────────
        answer = entry.get("answer", "")
        if answer:
            pdf.set_font("DejaVu", "B", 10)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 6, "Answer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("DejaVu", "", 9)
            pdf.multi_cell(0, 5, answer)
            pdf.ln(4)

        # ── Text chunks ────────────────────────────────────────────────────────
        chunks = entry.get("chunks", [])
        if chunks:
            pdf.set_font("DejaVu", "B", 10)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 6, f"Retrieved text chunks ({len(chunks)})",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
            for chunk in chunks:
                pdf.chunk_block(
                    score=chunk.get("score", 0.0),
                    pdf_id=chunk.get("pdf_id", ""),
                    text=chunk.get("text", ""),
                )
        else:
            pdf.set_font("DejaVu", "I", 9)
            pdf.set_text_color(160, 160, 160)
            pdf.cell(0, 6, "No chunks retrieved (score below threshold).",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)

        # ── Images ────────────────────────────────────────────────────────────
        images = entry.get("images", [])
        if images:
            pdf.ln(4)
            pdf.set_font("DejaVu", "B", 10)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 6, f"Retrieved page images ({len(images)})",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
            for img_path in images:
                pdf.embed_image(img_path, caption=Path(img_path).name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"Report written to {out_path}  ({len(entries)} queries)")

    # ── Clean up log and images after report is written ────────────────────────
    log_path.write_text("")
    print(f"Cleared {log_path}")
    img_dir = log_path.parent / "images"
    if img_dir.exists():
        removed = 0
        for f in img_dir.glob("*.png"):
            f.unlink()
            removed += 1
        print(f"Removed {removed} image(s) from {img_dir}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Build PDF report from retrieval log")
    ap.add_argument("--log", default=RETRIEVAL_LOG, help="Path to retrieval_log.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT,   help="Output PDF path")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        raise SystemExit(1)

    build_report(log_path, Path(args.out))


if __name__ == "__main__":
    _cli()
