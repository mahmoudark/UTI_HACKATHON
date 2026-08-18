def detect_intent(query):

    q = query.lower()

    antibiotic_terms = [
        "antibiotic",
        "antibiotics",
        "antimicrobial",
        "antimicrobials"
    ]

    selection_terms = [
        "prescrib",
        "choos",
        "select",
        "choice",
        "choosing",
        "selecting",
        "selection",
        "decision",
        "decisions",
        "treatment choice"
    ]

    consideration_terms = [
        "consider",
        "considered",
        "consideration",
        "considerations",
        "factor",
        "factors",
        "guide",
        "guiding",
        "influence",
        "influences",
        "take into account",
        "taken into account",
        "account",
        "important",
        "before prescribing"
    ]

    has_antibiotic = any(
        term in q
        for term in antibiotic_terms
    )

    has_selection = any(
        term in q
        for term in selection_terms
    )

    has_consideration = any(
        term in q
        for term in consideration_terms
    )

    # Strong antibiotic-selection intent
    if (
        has_antibiotic
        and has_selection
        and has_consideration
    ):
        return "ANTIBIOTIC_SELECTION"

    # Antibiotic prescribing/selection decisions
    # for lower UTI, even without explicit
    # consideration/factor wording.
    if (
        has_antibiotic
        and has_selection
        and (
            "lower uti" in q
            or "lower urinary tract infection" in q
        )
    ):
        return "ANTIBIOTIC_SELECTION"

    # Antibiotic + consideration/factor intent
    # for lower UTI.
    if (
        has_antibiotic
        and has_consideration
        and (
            "lower uti" in q
            or "lower urinary tract infection" in q
        )
    ):
        return "ANTIBIOTIC_SELECTION"

    # Patient-group / table intents

    if (
        "men aged 16 years and over" in q
        and "antibiotics" in q
    ):
        return "TABLE_3"

    if (
        "pregnant women aged 12 years and over" in q
        and "antibiotics" in q
    ):
        return "TABLE_2"

    if (
        "non-pregnant women aged 16 years and over" in q
        and "antibiotics" in q
    ):
        return "TABLE_1"

    if (
        "children and young people under 16 years" in q
        and "antibiotics" in q
    ):
        return "TABLE_4"

    return "GENERAL"


def intent_bonus(question, metadata, document):

    intent = detect_intent(question)

    source_id = None

    if isinstance(metadata, dict):
        source_id = (
            metadata.get("source_id")
            or metadata.get("id")
            or metadata.get("source")
        )

    if source_id is None and isinstance(document, dict):
        source_id = (
            document.get("source_id")
            or document.get("id")
            or document.get("source")
        )

    if (
        intent == "ANTIBIOTIC_SELECTION"
        and source_id == "1.4.1"
    ):
        return 0.10

    return 0.0