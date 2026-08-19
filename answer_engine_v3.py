import re
from ask import hybrid_search, generate_answer, TOP_K, SIMILARITY_THRESHOLD, LEXICAL_WEIGHT

POPULATION_TERMS = {
    "pregnant_women": ["pregnant women", "pregnant"],
    "non_pregnant_women": ["non-pregnant women", "nonpregnant women"],
    "men": ["men", "male"],
    "children": ["children", "young people", "under 16"],
}


def normalize(text):
    return re.sub(r"[^a-z0-9\s-]", " ", text.lower())


def detect_populations(text):
    t = normalize(text)
    found = set()

    # Non-pregnant women / woman
    if (
        "non-pregnant women" in t
        or "nonpregnant women" in t
        or "non-pregnant woman" in t
        or "nonpregnant woman" in t
    ):
        found.add("non_pregnant_women")

    # Pregnant women / woman
    elif (
        "pregnant women" in t
        or "pregnant woman" in t
        or "pregnant" in t
    ):
        found.add("pregnant_women")

    # Men / male / man
    if (
        "men" in t
        or "male" in t
        or "man" in t
    ):
        found.add("men")

    # Children / young people
    if (
        "children" in t
        or "child" in t
        or "young people" in t
        or "under 16" in t
    ):
        found.add("children")

    return found


def population_compatible(query, result):
    qpop = detect_populations(query)

    if not qpop:
        return True

    source_text = normalize(" ".join([
        str(result["metadata"].get("title", "")),
        str(result["metadata"].get("source_id", "")),
    ]))

    # Check NON-PREGNANT before PREGNANT because
    # "pregnant women" is a substring of "non-pregnant women".
    if "non-pregnant women" in source_text:
        return "non_pregnant_women" in qpop

    if "pregnant women" in source_text:
        return "pregnant_women" in qpop

    if "men aged 16 years and over" in source_text:
        return "men" in qpop

    if "children and young people under 16 years" in source_text:
        return "children" in qpop

    return True


def confidence_label(result, rank):
    sim = result["similarity"]
    hybrid = result["hybrid_score"]
    if rank == 1 and sim >= 0.50 and hybrid >= 0.55:
        return "HIGH"
    if rank <= 2 and sim >= 0.30 and hybrid >= 0.32:
        return "MEDIUM"
    if sim >= SIMILARITY_THRESHOLD:
        return "LOW"
    return "INSUFFICIENT"


def answer_supported(answer, evidence):
    answer_tokens = {x for x in re.findall(r"[a-zA-Z0-9]+", answer.lower()) if len(x) >= 4}
    evidence_tokens = {x for x in re.findall(r"[a-zA-Z0-9]+", evidence.lower()) if len(x) >= 4}
    if not answer_tokens:
        return True, 1.0
    coverage = len(answer_tokens & evidence_tokens) / len(answer_tokens)
    return coverage >= 0.55, coverage


def select_evidence(query, results):
    for rank, result in enumerate(results, start=1):
        if population_compatible(query, result):
            return result, rank
    return None, None


# ============================================================
# INPUT RISK CHECK
# Must run BEFORE retrieval.
# ============================================================

OUT_OF_SCOPE_MEDICAL_TERMS = [
    "diabetes",
    "pneumonia",
    "hypertension",
    "high blood pressure",
    "asthma",
    "appendicitis",
    "migraine",
    "heart attack",
    "influenza",
    "chronic kidney disease",
    "meningitis",
]

PERSONAL_ADVICE_PATTERNS = [
    "for me",
    "for myself",
    "my father",
    "my mother",
    "my child",
    "my son",
    "my daughter",
    "my husband",
    "my wife",
    "my patient",
    "should i",
    "can i take",
    "can i use",
    "is it safe for me",
    "what should i do",
    "do i need",
]


def input_risk_check(query):
    q = normalize(query)

    # Personal medical advice
    for pattern in PERSONAL_ADVICE_PATTERNS:
        pattern_re = r"\b" + re.escape(pattern) + r"\b"
        if re.search(pattern_re, q):
            return False, "Personal medical advice is outside the system scope."

    # Known out-of-scope medical topics.
    # Allow UTI-context questions to continue to retrieval;
    # unsupported combinations can still be refused by later gates.
    has_uti_context = (
        "uti" in q
        or "urinary tract infection" in q
        or "lower urinary tract infection" in q
        or "lower uti" in q
    )

    if not has_uti_context:
        for term in OUT_OF_SCOPE_MEDICAL_TERMS:
            if term in q:
                return False, "The question is outside the UTI guideline scope."

    return True, None

def answer_with_guard(query):

    # Check scope BEFORE retrieval.
    safe, risk_reason = input_risk_check(query)

    if not safe:
        return {
            "status": "REFUSED",
            "reason": risk_reason,
            "answer": None,
            "source": None,
            "rank": None,
            "confidence": "INSUFFICIENT",
            "grounding": None,
            "results": []
        }

    results = hybrid_search(query, top_k=TOP_K)
    if not results:
        return {
            "status": "REFUSED",
            "reason": "No evidence passed the similarity threshold.",
            "answer": None, "source": None, "rank": None,
            "confidence": "INSUFFICIENT", "grounding": None, "results": []
        }

    best, rank = select_evidence(query, results)
    if best is None:
        return {
            "status": "REFUSED",
            "reason": "Retrieved evidence did not match the requested patient population.",
            "answer": None, "source": None, "rank": None,
            "confidence": "INSUFFICIENT", "grounding": None, "results": results
        }

    confidence = confidence_label(best, rank)
    if confidence == "INSUFFICIENT":
        return {
            "status": "REFUSED",
            "reason": "Evidence confidence is insufficient.",
            "answer": None, "source": best["metadata"], "rank": rank,
            "confidence": confidence, "grounding": None, "results": results
        }

    answer = generate_answer(query, best)
    supported, coverage = answer_supported(answer, best["document"])
    if not supported:
        return {
            "status": "REFUSED",
            "reason": "The generated answer did not have enough lexical support in the retrieved evidence.",
            "answer": None, "source": best["metadata"], "rank": rank,
            "confidence": confidence, "grounding": coverage, "results": results
        }

    # Calibrate uncertainty language to evidence strength.
    if confidence == "HIGH":
        answer = "The guideline recommends:\n\n" + answer.strip()

    elif confidence == "MEDIUM":
        answer = "The guideline suggests:\n\n" + answer.strip()

    elif confidence == "LOW":
        answer = (
            "Limited evidence found; consider consulting the full guideline.\n\n"
            + answer.strip()
        )
    return {
        "status": "ANSWERED",
        "reason": "Evidence passed retrieval and validation checks.",
        "answer": answer, "source": best["metadata"], "rank": rank,
        "confidence": confidence,
        "evidence_match": best.get("evidence_match"),
        "grounding": coverage,
        "results": results
    }


def print_result(query, result):
    print("\n" + "=" * 70)
    print("ANSWER ENGINE V3")
    print("=" * 70)
    print("\nQUESTION\n" + query)
    print("\nSTATUS\n" + result["status"])
    print("\nREASON\n" + result["reason"])
    print("\nCONFIDENCE\n" + result["confidence"])
    print("\nRANK\n" + str(result["rank"]))
    if result["grounding"] is not None:
        print("\nGROUNDING COVERAGE:\n" + f"{result['grounding']:.4f}")
    if result["status"] == "ANSWERED":
        print("\nANSWER\n" + "=" * 70 + "\n" + result["answer"])
        m = result["source"]
        print("\nSOURCE\n" + "=" * 70)
        print("Source ID:", m.get("source_id"))
        print("Source type:", m.get("source_type"))
        print("Title:", m.get("title"))
        print("Page(s):", m.get("pages"))
    else:
        print("\nREFUSAL\n" + "=" * 70)
        print("The system will not provide a recommendation because the available evidence is insufficient or does not match the requested population.")


TEST_CASES = [
    "What should be done for pregnant women and men with lower UTI?",
    "What antibiotics are recommended for men aged 16 years and over?",
    "What should be considered when prescribing antibiotics for lower UTI?",
    "What is diabetes mellitus?",
]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("UTI CLINICAL DECISION SUPPORT - ANSWER ENGINE V3")
    print("=" * 70)
    print("Embedding:", "all-mpnet-base-v2")
    print("Top-K:", TOP_K)
    print("Similarity Threshold:", SIMILARITY_THRESHOLD)
    print("Lexical Weight:", LEXICAL_WEIGHT)
    print("\nRunning V3 safety tests...")
    for question in TEST_CASES:
        print_result(question, answer_with_guard(question))
    print("\n" + "=" * 70)
    print("V3 TESTING COMPLETED")
    print("=" * 70)






