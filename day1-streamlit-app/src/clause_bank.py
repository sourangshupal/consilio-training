"""Shared legal-domain example library used across every tab.

Centralized so Transformers 101, Tokenization, Model Landscape, and Prompt
Engineering all draw from the same curated set instead of duplicating strings.
"""

# --- Sentences for attention visualization (Transformers 101) --------------
ATTENTION_SENTENCES = {
    "Indemnification": (
        "The Lessee shall indemnify the Lessor against any claim arising "
        "therefrom, provided that it was not caused by gross negligence."
    ),
    "Termination": (
        "Either party may terminate this Agreement upon thirty days written "
        "notice if the other party materially breaches its obligations hereunder."
    ),
    "Confidentiality": (
        "The Receiving Party shall not disclose Confidential Information to any "
        "third party without the prior written consent of the Disclosing Party."
    ),
    "Arbitration": (
        "Any dispute arising out of or relating to this Agreement shall be "
        "resolved by binding arbitration administered by the AAA in New York."
    ),
    "Force Majeure": (
        "Neither party shall be liable for any failure to perform its "
        "obligations if such failure results from an event beyond its "
        "reasonable control."
    ),
    "Liability Cap": (
        "In no event shall either party's aggregate liability exceed the total "
        "fees paid under this Agreement in the twelve months preceding the claim."
    ),
}

# --- Toy corpus for hand-rolled BPE demo (Tokenization) --------------------
BPE_TOY_CORPUS = {
    "indemnify": 8,
    "indemnification": 5,
    "indemnitee": 3,
    "indemnitor": 2,
    "terminate": 6,
    "termination": 9,
    "terminated": 4,
}

# --- Legal / Latin terms for tokenizer shootout (Tokenization) -------------
SHOOTOUT_TERMS = [
    "force majeure", "res judicata", "estoppel", "indemnification",
    "notwithstanding", "hereinafter", "arbitration", "jurisdiction",
    "quantum meruit", "amicus curiae", "subpoena duces tecum", "habeas corpus",
    "voir dire", "ipso facto", "prima facie", "mens rea", "actus reus",
    "certiorari", "in camera", "pro bono", "caveat emptor", "de facto",
]

# --- Clauses for full-clause tokenizer comparison ---------------------------
COMPARISON_CLAUSES = {
    "Indemnification": ATTENTION_SENTENCES["Indemnification"],
    "Termination": ATTENTION_SENTENCES["Termination"],
    "Force Majeure": ATTENTION_SENTENCES["Force Majeure"],
}

# --- Clause-type taxonomy for zero-shot classification (Model Landscape /
#     Prompt Engineering) ----------------------------------------------------
CLAUSE_TAXONOMY = {
    "Indemnification": "A clause requiring one party to compensate the other for harm or loss.",
    "Termination": "A clause describing how and when the agreement may be ended.",
    "Confidentiality": "A clause restricting disclosure of confidential or proprietary information.",
    "Governing Law": "A clause specifying which jurisdiction's law governs the agreement.",
    "Force Majeure": "A clause excusing performance due to events beyond the parties' control.",
    "Arbitration": "A clause requiring disputes to be resolved via binding arbitration.",
    "Non-Compete": "A clause restricting a party from competing with the other after the relationship ends.",
    "Assignment": "A clause governing whether and how contractual rights may be transferred.",
    "Severability": "A clause providing that if one provision is invalid, the rest remain in effect.",
    "Dispute Resolution": "A clause outlining the process for resolving disagreements between the parties.",
}

CLASSIFY_TARGET_CLAUSES = {
    "Termination (from notebook)": ATTENTION_SENTENCES["Termination"],
    "Confidentiality": ATTENTION_SENTENCES["Confidentiality"],
    "Liability Cap": ATTENTION_SENTENCES["Liability Cap"],
}

# --- Fact patterns for CoT/IRAC and step-back prompting (Prompt Engineering) -
FACT_PATTERNS = {
    "Pedestrian Negligence": (
        "A pedestrian was struck by a delivery van while crossing at a "
        "marked crosswalk with the signal in her favor. The driver was "
        "checking a text message at the time of impact."
    ),
    "Breach of Contract": (
        "A software vendor agreed to deliver a custom CRM system by March 1. "
        "The vendor delivered a non-functional prototype on April 15, after "
        "the client had already paid 80% of the contract price."
    ),
    "Premises Liability": (
        "A grocery store customer slipped on a puddle near the produce "
        "section that had been reported to staff forty minutes earlier but "
        "was never cleaned up or marked with a warning sign."
    ),
}

# --- Personas for system-prompt comparison ----------------------------------
PERSONAS = {
    "Generic Assistant": "You are a helpful assistant.",
    "Contract-Review Paralegal": (
        "You are a meticulous paralegal reviewing contract clauses for "
        "potential risk to your client. Flag ambiguous language and "
        "one-sided terms."
    ),
    "Opposing Counsel": (
        "You are opposing counsel reviewing this clause for language that "
        "could be exploited or challenged in negotiation or litigation."
    ),
    "Compliance Officer": (
        "You are a corporate compliance officer checking this clause "
        "against regulatory and internal-policy risk."
    ),
}

SLIP_AND_FALL_SUMMARY = (
    "Case: Smith v. Jones Logistics. On June 3, plaintiff Maria Smith slipped "
    "on an unmarked wet floor near the loading dock of Jones Logistics' "
    "warehouse. Smith suffered a fractured wrist and missed six weeks of "
    "work. A warehouse employee stated the spill had been reported to a "
    "supervisor roughly one hour before the incident. No warning signs or "
    "cones were placed near the spill."
)

QWEN_PROMPT_PRESETS = {
    "Attorney Advertising Disclaimer": (
        "Write a short attorney-advertising disclaimer suitable for the "
        "footer of a law firm's website."
    ),
    "Plain-Language Clause Summary": (
        "Rewrite this clause in plain, client-friendly language: "
        + ATTENTION_SENTENCES["Liability Cap"]
    ),
    "Client Email Draft": (
        "Draft a short, professional email to a client explaining that their "
        "contract renewal deadline is in two weeks."
    ),
}
