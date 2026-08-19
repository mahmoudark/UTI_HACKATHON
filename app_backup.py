import streamlit as st
from ask import hybrid_search, generate_answer, MODEL_NAME, TOP_K, SIMILARITY_THRESHOLD, collection
st.set_page_config(
    page_title="UTI Clinical Decision Support",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 UTI Clinical Decision Support")
st.caption("Evidence-Grounded Clinical Question Answering")

st.divider()

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("⚙️ System Configuration")

    st.write("**Embedding Model**")
    st.code(MODEL_NAME)

    st.write("**Top-K**")
    st.code(str(TOP_K))

    st.write("**Similarity Threshold**")
    st.code(str(SIMILARITY_THRESHOLD))

    st.write("**Indexed Documents**")
    st.code(str(collection.count()))

    st.divider()

    st.success("✓ Grounded Retrieval")
    st.success("✓ Source Citation")
    st.success("✓ Refusal Logic")


# =========================
# CLINICAL ASSISTANT
# =========================

st.header("Clinical Assistant")

question = st.text_area(
    "Enter your clinical question",
    placeholder="Example: What antibiotics are recommended for non-pregnant women aged 16 years and over?",
    height=110
)

if st.button(
    "🔎 Get Evidence-Based Recommendation",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a clinical question.")

    else:

        with st.spinner("Retrieving clinical evidence..."):

            results = hybrid_search(
                question.strip(),
                top_k=TOP_K
            )

        # =========================
        # REFUSAL
        # =========================

        if not results:

            st.error("⚠️ Insufficient Evidence")

            st.info(
                "I couldn't find enough information in the indexed "
                "UTI guideline to answer this confidently. "
                "Please rephrase the question or consult a clinician directly."
            )

            st.caption(
                "Safety behavior: refusal triggered because "
                "no evidence passed the similarity threshold."
            )

        # =========================
        # ANSWER
        # =========================

        else:

            best = results[0]

            metadata = best["metadata"]

            answer = generate_answer(
                question.strip(),
                best
            )

            st.success("✓ Relevant clinical evidence found")

            st.header("💡 Recommendation")

            st.markdown(answer)

            st.divider()

            # =========================
            # METRICS
            # =========================

            st.header("📊 Retrieval Information")

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

            st.divider()

            # =========================
            # SOURCE
            # =========================

            st.header("📚 Source & Citation")

            c1, c2 = st.columns(2)

            with c1:

                st.write(
                    "**Source ID:**",
                    metadata.get("source_id")
                )

                st.write(
                    "**Source Type:**",
                    metadata.get("source_type")
                )

            with c2:

                st.write(
                    "**Title:**",
                    metadata.get("title")
                )

                st.write(
                    "**Page(s):**",
                    metadata.get("pages")
                )

            st.divider()

            # =========================
            # EVIDENCE
            # =========================

            st.header("🔎 Retrieved Evidence")

            st.info(
                best["document"]
            )

            # =========================
            # OTHER SOURCES
            # =========================

            with st.expander(
                "View all retrieved sources"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    meta = result["metadata"]

                    st.write(
                        f"**{i}. {meta.get('source_id')}**"
                    )

                    st.caption(
                        f"{meta.get('title')} | "
                        f"Similarity: "
                        f"{result['similarity']:.4f} | "
                        f"Hybrid: "
                        f"{result['hybrid_score']:.4f}"
                    )


# =========================
# PROJECT RESULTS
# =========================

st.divider()

st.header("📈 Project Results — Days 1–3")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Embedding",
    "MPNet"
)

col2.metric(
    "Recall@5",
    "100%"
)

col3.metric(
    "MRR",
    "0.9318"
)

col4.metric(
    "Hybrid Test",
    "5 / 5"
)


# =========================
# PIPELINE
# =========================

st.subheader("🔗 End-to-End Pipeline")

p1, p2, p3, p4, p5 = st.columns(5)

p1.info("1️⃣ Query\n\nClinical Question")

p2.info("2️⃣ Retrieve\n\nHybrid Search")

p3.info("3️⃣ Ground\n\nEvidence")

p4.info("4️⃣ Generate\n\nRecommendation")

p5.info("5️⃣ Cite / Refuse\n\nSafety")


# =========================
# SAFETY
# =========================

st.subheader("🛡️ Safety & Explainability")

s1, s2, s3 = st.columns(3)

s1.success("✓ Evidence Retrieval")
s2.success("✓ Citation Metadata")
s3.success("✓ Out-of-Scope Refusal")

st.caption(
    "UTI Clinical Decision Support — Hackathon Prototype"
)