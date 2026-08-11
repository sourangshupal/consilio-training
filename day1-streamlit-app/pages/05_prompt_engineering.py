import json
import re
from pathlib import Path

import streamlit as st

from src.clause_bank import (
    ATTENTION_SENTENCES,
    CLAUSE_TAXONOMY,
    FACT_PATTERNS,
    PERSONAS,
    SLIP_AND_FALL_SUMMARY,
)
from src.llm_router import LLMRouterError, ask
from src.state import active_api_key, init_session

init_session()

ASSETS = Path(__file__).parent.parent / "assets"

st.title("✍️ Legal Prompt Engineering Playground")
st.caption(":material/menu_book: Notebook: `06_legal_prompt_engineering_playground.ipynb`")

provider = st.session_state["provider"]
api_key = active_api_key()
has_key = bool(api_key)

if has_key:
    st.badge(f"Live via {provider}", icon=":material/bolt:", color="green")
else:
    st.badge("Static preview mode — add an API key in the sidebar to go live", icon=":material/wifi_off:", color="gray")


def run_live(label: str, prompt: str, system: str | None = None):
    """Renders a 'Run live' button; on click, calls the active provider and
    shows the result or a clean error."""
    if st.button(f"Run live: {label}", disabled=not has_key, key=f"btn_{label}", icon=":material/play_arrow:"):
        try:
            with st.spinner(f"Calling {provider}..."):
                result = ask(provider, api_key, prompt, system=system)
            st.success("Response:", icon=":material/chat:")
            st.write(result)
        except LLMRouterError as exc:
            st.error(str(exc))


col1, col2 = st.columns(2)
with col1:
    st.image(str(ASSETS / "11-prompt-engineering-slide04-anatomy-of-a-prompt.png"), caption="Anatomy of a prompt", width="stretch")
with col2:
    st.image(str(ASSETS / "12-prompt-engineering-slide15-decomposition-strategies.png"), caption="Decomposition strategies", width="stretch")

# --- Technique 0: Anatomy of a prompt ---------------------------------------
st.header(":material/looks_one: Anatomy of a prompt", divider="gray")
clause_pick = st.selectbox("Clause", list(ATTENTION_SENTENCES.keys()), key="anatomy_clause")
clause_text = ATTENTION_SENTENCES[clause_pick]
st.code(clause_text, language=None)

vague_prompt = f"What do you think of this: {clause_text}"
structured_prompt = (
    "Instruction: Review the following contract clause for one-sided or "
    "ambiguous language.\n"
    f"Context: The clause is from a commercial agreement.\n"
    f"Input: {clause_text}\n"
    "Output format: A bulleted list of concerns, each one sentence."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Vague prompt**")
    st.code(vague_prompt, language=None)
    run_live("Vague prompt", vague_prompt)
with c2:
    st.markdown("**Structured prompt (Instruction/Context/Input/Output-Format)**")
    st.code(structured_prompt, language=None)
    run_live("Structured prompt", structured_prompt)

# --- Technique 1: Zero-shot vs few-shot classification -----------------------
st.header(":material/looks_two: Zero-shot vs. few-shot clause classification", divider="gray")
labels_str = ", ".join(CLAUSE_TAXONOMY.keys())
target_clause = st.text_area(
    "Clause to classify",
    value=ATTENTION_SENTENCES["Confidentiality"],
    key="zs_clause",
)

zero_shot_prompt = (
    f"Classify the following clause into one of these categories: {labels_str}.\n"
    f"Clause: {target_clause}\nCategory:"
)
few_shot_prompt = (
    "Examples:\n"
    "Clause: 'This Agreement shall be governed by the laws of Delaware.' -> Governing Law\n"
    "Clause: 'Either party may terminate upon 30 days notice.' -> Termination\n\n"
    f"Classify the following clause into one of these categories: {labels_str}.\n"
    f"Clause: {target_clause}\nCategory:"
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Zero-shot**")
    st.code(zero_shot_prompt, language=None)
    run_live("Zero-shot classification", zero_shot_prompt)
with c2:
    st.markdown("**Few-shot**")
    st.code(few_shot_prompt, language=None)
    run_live("Few-shot classification", few_shot_prompt)

# --- Technique 2: CoT / IRAC --------------------------------------------------
st.header(":material/looks_3: Chain-of-thought / IRAC vs. direct answer", divider="gray")
fact_pick = st.selectbox("Fact pattern", list(FACT_PATTERNS.keys()), key="irac_fact")
fact_text = FACT_PATTERNS[fact_pick]
st.code(fact_text, language=None)

direct_prompt = f"{fact_text}\n\nWas the defendant negligent? Answer in one sentence."
irac_prompt = (
    f"{fact_text}\n\n"
    "Analyze using IRAC format (Issue, Rule, Application, Conclusion)."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Direct answer**")
    run_live("Direct answer", direct_prompt)
with c2:
    st.markdown("**IRAC reasoning**")
    run_live("IRAC reasoning", irac_prompt)

# --- Technique 3: Step-back prompting -----------------------------------------
st.header(":material/looks_4: Step-back prompting", divider="gray")
st.caption(f"Same fact pattern as above: {fact_pick}")
stepback_general = "What are the general elements required to prove negligence?"
stepback_applied = (
    f"General elements of negligence: duty, breach, causation, damages.\n\n"
    f"Apply these elements to the following facts: {fact_text}"
)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Step 1 — general question**")
    st.code(stepback_general, language=None)
    run_live("Step-back general", stepback_general)
with c2:
    st.markdown("**Step 2 — applied to facts**")
    st.code(stepback_applied, language=None)
    run_live("Step-back applied", stepback_applied)

# --- Technique 4: System prompt / persona -------------------------------------
st.header(":material/looks_5: System prompt / persona design", divider="gray")
persona_clause = st.selectbox("Clause to review", list(ATTENTION_SENTENCES.keys()), key="persona_clause")
persona_text = ATTENTION_SENTENCES[persona_clause]
persona_prompt = f"Review this clause: {persona_text}"

persona_cols = st.columns(len(PERSONAS))
for col, (name, system) in zip(persona_cols, PERSONAS.items()):
    with col:
        st.markdown(f"**{name}**")
        st.caption(system)
        run_live(f"Persona: {name}", persona_prompt, system=system)

# --- Technique 5: Structured output -------------------------------------------
st.header(":material/looks_6: Structured output", divider="gray")
st.subheader(":material/data_object: JSON-mode clause extraction")
json_clause = st.text_area(
    "Clause to extract metadata from",
    value=ATTENTION_SENTENCES["Indemnification"],
    key="json_clause",
)
json_prompt = (
    "Extract the following fields as JSON: clause_type, obligated_party, "
    "beneficiary_party, conditions, deadline_days (null if not specified).\n"
    f"Clause: {json_clause}\nRespond with ONLY valid JSON."
)
st.code(json_prompt, language=None)
if st.button("Run live: JSON extraction", disabled=not has_key, icon=":material/play_arrow:"):
    try:
        with st.spinner(f"Calling {provider}..."):
            result = ask(provider, api_key, json_prompt)
        cleaned = result.strip().strip("`").removeprefix("json").strip()
        try:
            st.json(json.loads(cleaned))
        except json.JSONDecodeError:
            st.warning("Response wasn't valid JSON — showing raw text.")
            st.write(result)
    except LLMRouterError as exc:
        st.error(str(exc))

st.subheader(":material/code: XML-tag reasoning/answer extraction")
xml_fact = st.text_area(
    "Dispute description",
    value=(
        "A commercial lease includes a CPI-indexed rent escalation clause. "
        "The landlord applied a 6% increase while the published CPI rose "
        "only 3.5% that year."
    ),
    key="xml_fact",
)
xml_prompt = (
    f"{xml_fact}\n\n"
    "Respond with <reasoning>your analysis</reasoning> and "
    "<answer>your conclusion in one sentence</answer>."
)
st.code(xml_prompt, language=None)
if st.button("Run live: XML extraction", disabled=not has_key, icon=":material/play_arrow:"):
    try:
        with st.spinner(f"Calling {provider}..."):
            result = ask(provider, api_key, xml_prompt)
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", result, re.DOTALL)
        answer = re.search(r"<answer>(.*?)</answer>", result, re.DOTALL)
        if reasoning:
            st.markdown("**Reasoning:**")
            st.write(reasoning.group(1).strip())
        if answer:
            st.markdown("**Answer:**")
            st.write(answer.group(1).strip())
        if not reasoning and not answer:
            st.warning("No XML tags found — showing raw response.")
            st.write(result)
    except LLMRouterError as exc:
        st.error(str(exc))

# --- Technique 6: Prompt chaining ---------------------------------------------
st.header(":material/link: Prompt chaining (3-stage pipeline)", divider="gray")
case_summary = st.text_area("Case summary", value=SLIP_AND_FALL_SUMMARY, height=100, key="chain_case")

if st.button("Run live: Full 3-stage chain", disabled=not has_key, icon=":material/play_arrow:"):
    try:
        with st.spinner(f"Stage 1/3 — extracting facts via {provider}..."):
            facts = ask(provider, api_key, f"Extract the key facts as a bulleted list:\n{case_summary}")
        st.markdown("**Stage 1 — Extracted facts:**")
        st.write(facts)

        with st.spinner("Stage 2/3 — drafting Statement of Facts..."):
            draft = ask(
                provider, api_key,
                f"Using these facts, draft a formal 'Statement of Facts' memo section:\n{facts}",
            )
        st.markdown("**Stage 2 — Drafted memo section:**")
        st.write(draft)

        with st.spinner("Stage 3/3 — validating against source..."):
            validation = ask(
                provider, api_key,
                f"Original case summary:\n{case_summary}\n\nDrafted memo section:\n{draft}\n\n"
                "Does the drafted section accurately reflect the original summary? "
                "Flag any unsupported claims.",
            )
        st.markdown("**Stage 3 — Validation:**")
        st.write(validation)
    except LLMRouterError as exc:
        st.error(str(exc))
