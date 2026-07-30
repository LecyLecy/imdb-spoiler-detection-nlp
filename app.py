from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


APP_TITLE = "PlotGuard — Movie Spoiler Detector"
SPOILER_THRESHOLD = 0.50
SAFE_EXAMPLE = (
    "Beautiful cinematography, a memorable score, and strong performances make "
    "this an easy recommendation for science-fiction fans."
)
SPOILER_EXAMPLE = (
    "In the final scene, the detective discovers that his partner planned the "
    "entire crime and was the real villain all along."
)


@st.cache_resource(show_spinner=False)
def load_components() -> tuple[Any, str]:
    """Load the trained pipeline and its display name once per app session."""
    model_directory = Path(__file__).resolve().parent / "models"
    pipeline_path = model_directory / "spoiler_detection_pipeline.pkl"
    model_name_path = model_directory / "best_model_name.pkl"

    pipeline = joblib.load(pipeline_path)
    model_name = str(joblib.load(model_name_path))
    return pipeline, model_name


def predict_spoiler(review_text: str, pipeline: Any) -> dict[str, float | int | bool]:
    """Return a spoiler prediction using the feature schema used for training."""
    word_count = len(review_text.split())
    input_frame = pd.DataFrame(
        {"review_text": [review_text], "word_count": [word_count]}
    )
    probabilities = pipeline.predict_proba(input_frame)[0]

    classes = list(getattr(pipeline, "classes_", []))
    spoiler_index = next(
        (
            index
            for index, label in enumerate(classes)
            if label is True or label == 1 or str(label).lower() == "spoiler"
        ),
        1,
    )
    spoiler_probability = float(probabilities[spoiler_index])

    return {
        "is_spoiler": spoiler_probability >= SPOILER_THRESHOLD,
        "spoiler_probability": spoiler_probability,
        "safe_probability": 1.0 - spoiler_probability,
        "word_count": word_count,
    }


def set_review_example(review_text: str) -> None:
    """Replace the current input with an example and discard any stale result."""
    st.session_state["review_input"] = review_text
    st.session_state.pop("analysis_result", None)


def clear_review() -> None:
    """Clear the workspace."""
    st.session_state["review_input"] = ""
    st.session_state.pop("analysis_result", None)


def inject_styles() -> None:
    """Apply the portfolio UI theme."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #f7f8fc;
            --muted: #a8afc1;
            --panel: rgba(18, 22, 34, 0.84);
            --panel-strong: #151a28;
            --line: rgba(255, 255, 255, 0.10);
            --accent: #ff5d73;
            --accent-soft: rgba(255, 93, 115, 0.14);
            --safe: #51d6a2;
            --safe-soft: rgba(81, 214, 162, 0.13);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 0%, rgba(255, 93, 115, 0.13), transparent 32rem),
                radial-gradient(circle at 92% 22%, rgba(102, 93, 255, 0.11), transparent 28rem),
                #090c14;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 3.5rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, p, label, .stMarkdown {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .hero {
            max-width: 820px;
            margin-bottom: 2.25rem;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1.15rem;
            color: #c8cede;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .eyebrow-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 6px var(--accent-soft);
        }

        .hero h1 {
            margin: 0;
            color: var(--ink);
            font-size: clamp(2.7rem, 6vw, 4.7rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
            font-weight: 780;
        }

        .hero h1 .accent-line {
            color: var(--accent);
        }

        .hero p {
            max-width: 690px;
            margin: 1.35rem 0 0;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(23, 28, 43, 0.93), rgba(14, 18, 29, 0.9));
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
        }

        [data-testid="stTextArea"] textarea {
            min-height: 218px;
            padding: 1rem 1.05rem;
            color: var(--ink);
            background: rgba(6, 9, 16, 0.62);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            line-height: 1.65;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(255, 93, 115, 0.78);
            box-shadow: 0 0 0 3px rgba(255, 93, 115, 0.11);
        }

        .stButton > button {
            min-height: 2.8rem;
            border-radius: 11px;
            border-color: rgba(255, 255, 255, 0.13);
            font-weight: 700;
            transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255, 255, 255, 0.28);
        }

        .stButton > button[kind="primary"] {
            color: #ffffff;
            background: linear-gradient(135deg, #ff5d73, #e94264);
            border: 0;
            box-shadow: 0 10px 24px rgba(233, 66, 100, 0.25);
        }

        .section-label {
            margin: 0 0 0.35rem;
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .section-copy {
            margin: 0 0 1rem;
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.55;
        }

        .result-card {
            min-height: 304px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0.2rem 0.1rem 0.25rem;
        }

        .status-badge {
            display: inline-flex;
            width: fit-content;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.68rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .status-badge.spoiler {
            color: #ff8b9a;
            background: var(--accent-soft);
            border: 1px solid rgba(255, 93, 115, 0.28);
        }

        .status-badge.safe {
            color: #7ee7bf;
            background: var(--safe-soft);
            border: 1px solid rgba(81, 214, 162, 0.25);
        }

        .result-title {
            margin: 1.15rem 0 0.55rem;
            color: var(--ink);
            font-size: 1.85rem;
            line-height: 1.12;
            letter-spacing: -0.035em;
        }

        .result-copy {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .probability-row {
            display: flex;
            justify-content: space-between;
            margin: 1.4rem 0 0.55rem;
            color: #c9cede;
            font-size: 0.82rem;
            font-weight: 650;
        }

        .probability-row strong {
            color: var(--ink);
        }

        .score-track {
            height: 9px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 99px;
        }

        .score-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #ff7185, #ff425f);
        }

        .empty-state {
            min-height: 304px;
            display: grid;
            place-items: center;
            padding: 1.5rem;
            text-align: center;
        }

        .empty-icon {
            width: 60px;
            height: 60px;
            display: grid;
            place-items: center;
            margin: 0 auto 1.2rem;
            color: #ff8191;
            background: var(--accent-soft);
            border: 1px solid rgba(255, 93, 115, 0.24);
            border-radius: 18px;
            font-size: 1.65rem;
        }

        .empty-state h3 {
            margin-bottom: 0.45rem;
            color: var(--ink);
            font-size: 1.2rem;
        }

        .empty-state p {
            max-width: 320px;
            margin: 0 auto;
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.9rem;
        }

        .workflow {
            margin-top: 3.2rem;
            padding-top: 2.2rem;
            border-top: 1px solid var(--line);
        }

        .workflow-heading {
            margin-bottom: 1.25rem;
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.025em;
        }

        .step-card {
            min-height: 138px;
            padding: 1.1rem 1.15rem;
            background: rgba(16, 20, 31, 0.58);
            border: 1px solid var(--line);
            border-radius: 15px;
        }

        .step-number {
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.11em;
        }

        .step-card h3 {
            margin: 0.55rem 0 0.4rem;
            color: var(--ink);
            font-size: 1rem;
        }

        .step-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .footer-note {
            margin-top: 2.3rem;
            color: #767e91;
            font-size: 0.78rem;
            text-align: center;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 2.2rem;
            }

            .hero h1 {
                font-size: 2.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any]) -> None:
    """Render a human-readable classifier result."""
    is_spoiler = bool(result["is_spoiler"])
    spoiler_probability = float(result["spoiler_probability"])
    status_class = "spoiler" if is_spoiler else "safe"
    status_label = "Potential spoiler" if is_spoiler else "Likely safe"
    title = "Plot details may be revealed." if is_spoiler else "This looks safe to read."
    description = (
        "The wording contains signals commonly associated with important plot "
        "events, reveals, or endings."
        if is_spoiler
        else "The review appears to focus on general opinions rather than specific "
        "story events or reveals."
    )

    st.markdown(
        f"""
        <div class="result-card" role="status">
            <div>
                <div class="status-badge {status_class}">{status_label}</div>
                <h2 class="result-title">{title}</h2>
                <p class="result-copy">{description}</p>
            </div>
            <div>
                <div class="probability-row">
                    <span>Spoiler likelihood</span>
                    <strong>{spoiler_probability:.0%}</strong>
                </div>
                <div class="score-track" aria-label="Spoiler likelihood">
                    <div class="score-fill" style="width: {spoiler_probability:.2%};"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎞️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    try:
        pipeline, model_name = load_components()
        model_error = None
    except FileNotFoundError as error:
        pipeline, model_name = None, "Unavailable"
        model_error = f"Required model file not found: {error.filename}"
    except Exception as error:
        pipeline, model_name = None, "Unavailable"
        model_error = f"The model could not be loaded: {error}"

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">
                <span class="eyebrow-dot"></span>
                NLP portfolio project
            </div>
            <h1>Know before you read.<br><span class="accent-line">Keep the plot intact.</span></h1>
            <p>
                PlotGuard uses a trained natural-language classifier to estimate
                whether a movie review reveals meaningful story details.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_column, result_column = st.columns([1.14, 0.86], gap="large")

    with input_column:
        with st.container(border=True):
            st.markdown('<p class="section-label">Review workspace</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-copy">Paste a review, or load an example to see the classifier in action.</p>',
                unsafe_allow_html=True,
            )

            example_safe, example_spoiler = st.columns(2)
            with example_safe:
                st.button(
                    "Load a safe example",
                    use_container_width=True,
                    on_click=set_review_example,
                    args=(SAFE_EXAMPLE,),
                )
            with example_spoiler:
                st.button(
                    "Load a spoiler example",
                    use_container_width=True,
                    on_click=set_review_example,
                    args=(SPOILER_EXAMPLE,),
                )

            review_text = st.text_area(
                "Movie review",
                key="review_input",
                label_visibility="collapsed",
                placeholder=(
                    "Paste an IMDb-style review here. For example: The performances "
                    "are excellent, but the final reveal changes everything..."
                ),
            )

            word_count = len(review_text.split())
            st.caption(f"{word_count:,} words · English-language reviews work best")

            analyze_column, clear_column = st.columns([1.55, 0.75])
            with analyze_column:
                analyze_clicked = st.button(
                    "Analyze review",
                    type="primary",
                    use_container_width=True,
                    disabled=pipeline is None,
                )
            with clear_column:
                st.button(
                    "Clear",
                    use_container_width=True,
                    on_click=clear_review,
                    disabled=not review_text,
                )

            if analyze_clicked:
                if not review_text.strip():
                    st.warning("Enter a review before running the analysis.")
                else:
                    with st.spinner("Reading between the lines..."):
                        try:
                            prediction = predict_spoiler(review_text, pipeline)
                            st.session_state["analysis_result"] = {
                                **prediction,
                                "source_text": review_text,
                            }
                        except Exception:
                            st.session_state.pop("analysis_result", None)
                            st.error(
                                "The review could not be analyzed. Please try a different passage."
                            )

            if model_error:
                st.error("The classifier is temporarily unavailable.")
                with st.expander("Technical details"):
                    st.code(model_error)

    with result_column:
        with st.container(border=True):
            st.markdown('<p class="section-label">Analysis</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-copy">A probability estimate, translated into a clear reading recommendation.</p>',
                unsafe_allow_html=True,
            )

            result = st.session_state.get("analysis_result")
            if result and result.get("source_text") == review_text:
                render_result(result)
                with st.expander("Model details"):
                    st.write(f"**Classifier:** {model_name}")
                    st.write(f"**Decision threshold:** {SPOILER_THRESHOLD:.0%}")
                    st.write(f"**Review length:** {int(result['word_count']):,} words")
                    st.caption(
                        "This is an experimental machine-learning estimate, not a "
                        "guarantee. Ambiguous or very short reviews may be misclassified."
                    )
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                        <div>
                            <div class="empty-icon">◎</div>
                            <h3>Your result will appear here</h3>
                            <p>
                                Add a review and select <strong>Analyze review</strong>
                                to see its spoiler likelihood.
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<section class="workflow"><p class="workflow-heading">From review to recommendation</p></section>',
        unsafe_allow_html=True,
    )
    step_one, step_two, step_three = st.columns(3, gap="medium")
    steps = (
        (
            step_one,
            "01 · INPUT",
            "Read the review",
            "The original review text and its word count become the model inputs.",
        ),
        (
            step_two,
            "02 · FEATURES",
            "Map the language",
            "TF–IDF converts meaningful unigrams and bigrams into numeric features.",
        ),
        (
            step_three,
            "03 · PREDICTION",
            "Estimate spoiler risk",
            "A trained linear classifier returns the likelihood of spoiler content.",
        ),
    )
    for column, number, heading, copy in steps:
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <span class="step-number">{number}</span>
                    <h3>{heading}</h3>
                    <p>{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <p class="footer-note">
            Built with Streamlit, pandas, scikit-learn, and a healthy respect for plot twists.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
