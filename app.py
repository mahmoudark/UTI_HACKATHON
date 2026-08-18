import streamlit as st

from ask import (
    MODEL_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LEXICAL_WEIGHT,
    collection,
)

from answer_engine_v3 import answer_with_guard


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UTI Clinical Decision Support",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 UTI Clinical Decision Support")

st.caption(
    "Evidence-Grounded Clinical Question Answering"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    st.write("**Embedding Model**")
    st.code(MODEL_NAME)

    st.write("**Top-K**")
    st.code(str(TOP_K))

    st.write("**Similarity Threshold**")
    st.code(str(SIMILARITY_THRESHOLD))

    st.write("**Lexical Weight**")
    st.code(str(LEXICAL_WEIGHT))

    st.write("**Intent Weight**")
    st.code("0.10")

    st.write("**Indexed Documents**")
    st.code(str(collection.count()))

    st.divider()

    st.success("✓ Hybrid Retrieval")
    st.success("✓ Intent-Aware Retrieval")
    st.success("✓ Population Guard")
    st.success("✓ Evidence Citation")
    st.success("✓ Confidence Scoring")
    st.success("✓ Refusal Logic")


# ============================================================
# CLINICAL ASSISTANT
# ============================================================

st.header("Clinical Assistant")

question = st.text_area(
    "Enter your clinical question",
    placeholder=(
        "Example: What antibiotics are recommended "
        "for men aged 16 years and over?"
    ),
    height=120
)


if st.button(
    "🔎 Get Evidence-Based Recommendation",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a clinical question."
        )

    else:

        query = question.strip()

        with st.spinner(
            "Retrieving and validating clinical evidence..."
        ):

            result = answer_with_guard(query)


        # ====================================================
        # RESULT STATUS
        # ====================================================

        status = result["status"]
        confidence = result["confidence"]
        rank = result["rank"]
        grounding = result["grounding"]
        reason = result["reason"]


        # ====================================================
        # REFUSED
        # ====================================================

        if status == "REFUSED":

            st.error(
                "⚠️ Insufficient Evidence"
            )

            st.write(
                reason
            )

            st.info(
                "The system will not provide a clinical "
                "recommendation because the available "
                "evidence is insufficient or does not match "
                "the requested patient population."
            )

            # ------------------------------------------------
            # Retrieval information if available
            # ------------------------------------------------

            results = result.get("results", [])

            if results:

                st.divider()

                st.header(
                    "📊 Retrieval Information"
                )

                best = results[0]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Best Similarity",
                    f"{best['similarity']:.4f}"
                )

                col2.metric(
                    "Best Hybrid Score",
                    f"{best['hybrid_score']:.4f}"
                )

                col3.metric(
                    "Retrieved",
                    len(results)
                )


        # ====================================================
        # ANSWERED
        # ====================================================

        else:

            answer = result["answer"]
            source = result["source"]

            # ------------------------------------------------
            # SUCCESS STATUS
            # ------------------------------------------------

            st.success(
                "✓ Relevant clinical evidence found and validated"
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            st.header(
                "🎯 Confidence"
            )

            if confidence == "HIGH":

                st.success(
                    f"HIGH — Evidence rank: {rank}"
                )

            elif confidence == "MEDIUM":

                st.warning(
                    f"MEDIUM — Evidence rank: {rank}"
                )

            else:

                st.info(
                    f"{confidence} — Evidence rank: {rank}"
                )


            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.header(
                "💡 Recommendation"
            )

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # VALIDATION INFORMATION
            # ------------------------------------------------

            st.divider()

            st.header(
                "🛡️ Evidence Validation"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Confidence",
                confidence
            )

            col2.metric(
                "Evidence Rank",
                rank
            )

            if grounding is not None:

                col3.metric(
                    "Grounding Coverage",
                    f"{grounding:.2%}"
                )

            else:

                col3.metric(
                    "Grounding Coverage",
                    "N/A"
                )


            # ------------------------------------------------
            # SOURCE
            # ------------------------------------------------

            st.divider()

            st.header(
                "📚 Source & Citation"
            )

            if source:

                c1, c2 = st.columns(2)

                with c1:

                    st.write(
                        "**Source ID:**",
                        source.get("source_id", "N/A")
                    )

                    st.write(
                        "**Source Type:**",
                        source.get("source_type", "N/A")
                    )

                with c2:

                    st.write(
                        "**Title:**",
                        source.get("title", "N/A")
                    )

                    st.write(
                        "**Page(s):**",
                        source.get("pages", "N/A")
                    )


            # ------------------------------------------------
            # RETRIEVAL DETAILS
            # ------------------------------------------------

            results = result.get(
                "results",
                []
            )

            if results:

                st.divider()

                st.header(
                    "📊 Retrieval Details"
                )

                best = results[0]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Similarity",
                    f"{best['similarity']:.4f}"
                )

                col2.metric(
                    "Hybrid Score",
                    f"{best['hybrid_score']:.4f}"
                )

                col3.metric(
                    "Retrieved",
                    len(results)
                )

                col4.metric(
                    "Top-K",
                    TOP_K
                )


                # ------------------------------------------------
                # TOP RETRIEVED EVIDENCE
                # ------------------------------------------------

                with st.expander(
                    "View retrieved evidence"
                ):

                    for i, item in enumerate(
                        results,
                        start=1
                    ):

                        metadata = item["metadata"]

                        st.write(
                            f"### Rank {i}"
                        )

                        st.write(
                            "**Source ID:**",
                            metadata.get(
                                "source_id",
                                "N/A"
                            )
                        )

                        st.write(
                            "**Title:**",
                            metadata.get(
                                "title",
                                "N/A"
                            )
                        )

                        st.write(
                            "**Similarity:**",
                            f"{item['similarity']:.4f}"
                        )

                        st.write(
                            "**Hybrid Score:**",
                            f"{item['hybrid_score']:.4f}"
                        )

                        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UTI Clinical Decision Support — "
    "Evidence-Grounded Retrieval System"
)