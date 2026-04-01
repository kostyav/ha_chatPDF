"""Run 5 benchmark questions through the Part 4 SSE API, score with BERTScore."""
import json
import sys
import urllib.request

from bert_score import score as bert_score
from tabulate import tabulate

API = "http://localhost:8080/chat"

QUESTIONS = [
    "Schema of ester from ethyl chloroformate.",
    "Provide the numeric values and description of the antimicrobial activity of esters.",
    "Short description of bhutan",
    "Describe the smallpox vaccination program",
    "How the LLM spaceforce can help brazilian government to reduce the public debt",
]

# Reference answers used as BERTScore ground-truth.
# For Q1/Q2/Q4 these are grounded in the indexed PDFs (chemistry + smallpox docs).
# For Q3 (Bhutan geography) and Q5 (out-of-domain) the references reflect the
# expected correct answer so the score measures how well the agent did.
REFERENCES = [
    # Q1 — ester synthesis from ethyl chloroformate
    (
        "Ethyl chloroformate reacts with a ketone in the presence of a base to form "
        "a beta-keto ester intermediate. Addition of hydrazine or its derivatives to "
        "this intermediate yields pyrazolone products in situ."
    ),
    # Q2 — antimicrobial activity of esters (qualitative only in the docs)
    (
        "The antimicrobial activity of the ester compounds is described qualitatively "
        "as very good, moderate to good, moderate, or poor. No specific numeric MIC "
        "values are reported in the available documents."
    ),
    # Q3 — Bhutan (general knowledge)
    (
        "Bhutan is a landlocked kingdom in the Eastern Himalayas, bordered by India "
        "and China. It is known for its Gross National Happiness philosophy, Buddhist "
        "culture, and pristine mountain landscapes."
    ),
    # Q4 — smallpox vaccination programme (from indexed WHO/Bhutan docs)
    (
        "The WHO-led smallpox eradication programme gained momentum in the late 1960s. "
        "In Bhutan, Claude Smith led vaccination efforts from 1966, integrating them "
        "into new healthcare structures. Nineteen vaccinators were added in 1964 and "
        "five more in December 1965. Mass campaigns used a fixed-rigid needle technique. "
        "WHO SEARO reported on the Bhutanese programme in 1977."
    ),
    # Q5 — out-of-domain / no relevant documents
    (
        "The documents do not contain information about LLM spaceforce or Brazilian "
        "public debt reduction strategies."
    ),
]


def ask(question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    req  = urllib.request.Request(API, data=body,
                                  headers={"Content-Type": "application/json"})
    statuses, tokens = [], []
    with urllib.request.urlopen(req, timeout=300) as resp:
        event = None
        for raw in resp:
            line = raw.decode().rstrip("\n\r")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "status":
                    statuses.append(data)
                elif event == "token":
                    tokens.append(data)
    return {"statuses": statuses, "answer": "".join(tokens)}


def main():
    sep = "=" * 72
    results = []

    for i, (q, ref) in enumerate(zip(QUESTIONS, REFERENCES), 1):
        print(f"\n{sep}")
        print(f"Q{i}: {q}")
        print(sep)
        r = ask(q)
        for s in r["statuses"]:
            print(f"  [trace] {s}")
        print(f"\nAnswer:\n{r['answer']}\n")
        sys.stdout.flush()
        results.append({"q": q, "answer": r["answer"], "ref": ref,
                        "statuses": r["statuses"]})

    # ── BERTScore ──────────────────────────────────────────────────────────────
    print("\nComputing BERTScore (roberta-large) …")
    hypotheses = [r["answer"] for r in results]
    references  = [r["ref"]    for r in results]
    P, R, F1 = bert_score(hypotheses, references, lang="en",
                          model_type="roberta-large", verbose=False)

    for i, r in enumerate(results):
        r["bert_P"]  = float(P[i])
        r["bert_R"]  = float(R[i])
        r["bert_F1"] = float(F1[i])

    # ── Report table ───────────────────────────────────────────────────────────
    routed_via = [" → ".join(r["statuses"]) for r in results]
    rows = [
        [f"Q{i+1}", r["q"][:55] + ("…" if len(r["q"]) > 55 else ""),
         routed_via[i], f"{r['bert_P']:.4f}", f"{r['bert_R']:.4f}", f"{r['bert_F1']:.4f}"]
        for i, r in enumerate(results)
    ]
    headers = ["#", "Question", "Agent route", "BERTScore P", "BERTScore R", "BERTScore F1"]

    avg_f1 = float(F1.mean())
    print("\n" + "=" * 72)
    print("BERT SCORE REPORT — Part 4 Agent (gemma3:4b via Ollama)")
    print("=" * 72)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print(f"\nAverage BERTScore F1 : {avg_f1:.4f}")

    with open("src/part5_2/bert_results.json", "w") as f:
        json.dump({"avg_bert_f1": avg_f1, "results": results}, f, indent=2)
    print("Raw results saved → src/part5_2/bert_results.json")


if __name__ == "__main__":
    main()
