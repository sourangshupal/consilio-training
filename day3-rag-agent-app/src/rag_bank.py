"""Shared example library for the Day 3 RAG & Agents app: bundled sample
legal documents, per-page example queries, the RAGAS eval question bank, and
the agent tool's clause database. Centralized so every page draws from the
same curated set instead of duplicating strings.
"""

SERVICE_AGREEMENT = """SERVICE AGREEMENT

1. PARTIES
This Service Agreement ("Agreement") is entered into between Acme Corp ("Provider") and Beta Ltd ("Client").

2. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles.

3. TERM AND TERMINATION
3.1 This Agreement commences on the Effective Date and continues for a period of twelve (12) months.
3.2 Either party may terminate this Agreement upon thirty (30) days written notice.
3.3 Either party may terminate immediately if the other party commits a material breach that remains uncured for fifteen (15) days after written notice.

4. CONFIDENTIALITY
4.1 Each party agrees to maintain the confidentiality of all proprietary information disclosed by the other party.
4.2 Confidentiality obligations survive termination for a period of five (5) years.

5. INDEMNIFICATION
Provider agrees to indemnify and hold harmless Client from any claims arising from Provider's negligence or willful misconduct.

6. LIMITATION OF LIABILITY
Neither party shall be liable for indirect, incidental, or consequential damages. The total liability of either party under this Agreement shall not exceed the fees paid in the preceding six (6) months.

7. DISPUTE RESOLUTION
Any dispute arising from this Agreement shall be resolved through binding arbitration in Wilmington, Delaware.

8. INTELLECTUAL PROPERTY
All intellectual property created during the course of this Agreement shall remain the sole property of the creating party unless otherwise agreed in writing.
"""

NDA = """MUTUAL NON-DISCLOSURE AGREEMENT

1. PARTIES
This Mutual Non-Disclosure Agreement ("Agreement") is entered into between Northwind Industries ("Disclosing Party") and Contoso Labs ("Receiving Party").

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any technical, business, or financial information disclosed by either party, whether in written, oral, or electronic form, that is designated as confidential or that reasonably should be understood to be confidential.

3. OBLIGATIONS
3.1 The Receiving Party shall use the Confidential Information solely for the purpose of evaluating a potential business relationship.
3.2 The Receiving Party shall not disclose Confidential Information to any third party without the prior written consent of the Disclosing Party.
3.3 The Receiving Party shall protect the Confidential Information using the same degree of care it uses for its own confidential information, but no less than reasonable care.

4. EXCLUSIONS
Confidential Information does not include information that: (a) is or becomes publicly available through no fault of the Receiving Party; (b) was already known to the Receiving Party prior to disclosure; or (c) is independently developed without use of the Confidential Information.

5. TERM
5.1 This Agreement shall remain in effect for two (2) years from the Effective Date.
5.2 The confidentiality obligations under Section 3 shall survive termination of this Agreement for a period of three (3) years.

6. REMEDIES
The parties acknowledge that unauthorized disclosure may cause irreparable harm, entitling the Disclosing Party to seek injunctive relief in addition to any other remedies available at law.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.
"""

EMPLOYMENT_AGREEMENT = """EMPLOYMENT AGREEMENT

1. PARTIES AND POSITION
This Employment Agreement ("Agreement") is entered into between Meridian Analytics, Inc. ("Company") and the undersigned employee ("Employee") for the position of Senior Data Scientist.

2. COMPENSATION
2.1 Employee shall receive an annual base salary of $145,000, payable in accordance with the Company's standard payroll practices.
2.2 Employee is eligible for an annual discretionary bonus of up to 15% of base salary, subject to Company performance and individual review.

3. AT-WILL EMPLOYMENT
Employment under this Agreement is at-will. Either party may terminate the employment relationship at any time, with or without cause, subject to the notice provisions in Section 4.

4. TERMINATION AND NOTICE
4.1 The Company shall provide two (2) weeks' written notice, or pay in lieu thereof, in the event of termination without cause.
4.2 The Employee shall provide two (2) weeks' written notice of resignation.
4.3 The Company may terminate immediately for cause, including gross misconduct, breach of confidentiality, or material violation of Company policy.

5. NON-COMPETE AND NON-SOLICITATION
5.1 For a period of twelve (12) months following termination, Employee shall not provide competing services to a direct competitor of the Company within the United States.
5.2 For a period of twelve (12) months following termination, Employee shall not solicit Company clients or employees.

6. CONFIDENTIALITY AND INVENTIONS
All work product, inventions, and proprietary information developed during employment shall be the sole property of the Company.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York.
"""

COMMERCIAL_LEASE = """COMMERCIAL LEASE AGREEMENT

1. PARTIES AND PREMISES
This Commercial Lease Agreement ("Lease") is entered into between Harborview Properties LLC ("Landlord") and Fulcrum Consulting Group ("Tenant") for the premises located at 400 Market Street, Suite 210.

2. TERM
2.1 The initial term of this Lease shall be five (5) years, commencing on the Lease Commencement Date.
2.2 Tenant shall have the option to renew for one additional five (5) year term, provided written notice is given at least one hundred eighty (180) days prior to expiration.

3. RENT
3.1 Base rent shall be $8,500 per month, subject to an annual escalation of three percent (3%) or the increase in the Consumer Price Index, whichever is greater.
3.2 Tenant shall pay its proportionate share of common area maintenance charges, real estate taxes, and building insurance.

4. USE OF PREMISES
The premises shall be used solely for general office purposes and shall not be used for any purpose that violates applicable zoning ordinances.

5. MAINTENANCE AND REPAIRS
5.1 Landlord shall maintain the structural elements, roof, and exterior of the building.
5.2 Tenant shall maintain the interior of the premises in good condition, ordinary wear and tear excepted.

6. DEFAULT AND REMEDIES
6.1 Failure to pay rent within ten (10) days of the due date constitutes a default.
6.2 Landlord may terminate this Lease and pursue all available remedies if a default remains uncured for thirty (30) days after written notice.

7. GOVERNING LAW
This Lease shall be governed by the laws of the State of Illinois.
"""


def sample_documents() -> list[dict]:
    """Bundled sample corpus, in the same {"text","filename","type"} shape
    src/loader.py produces, so it plugs directly into chunk_documents()."""
    return [
        {"text": SERVICE_AGREEMENT, "filename": "service_agreement.md", "type": "markdown"},
        {"text": NDA, "filename": "mutual_nda.md", "type": "markdown"},
        {"text": EMPLOYMENT_AGREEMENT, "filename": "employment_agreement.md", "type": "markdown"},
        {"text": COMMERCIAL_LEASE, "filename": "commercial_lease.md", "type": "markdown"},
    ]


# --- Example queries per technique, for dropdowns alongside free text -------
VANILLA_RAG_QUERIES = [
    "What is the governing law of the service agreement?",
    "How much notice is required to terminate the employment agreement?",
    "What is the annual rent escalation in the commercial lease?",
    "What are the confidentiality obligations in the NDA?",
    "What is the liability cap in the service agreement?",
    "Can the employee compete with the company after leaving?",
]

BM25_VS_DENSE_QUERIES = {
    "Exact term match (BM25 should win)": "material breach",
    "Exact section reference (BM25 should win)": "Section 4.3",
    "Paraphrase — contract violation (dense should win)": "What happens if someone breaks the contract badly?",
    "Paraphrase — rent over time (dense should win)": "How is rent priced over time?",
}

HYBRID_RERANK_QUERIES = [
    "What are the termination conditions across these documents?",
    "How is confidential information protected?",
    "What happens if a party defaults?",
    "What are the notice periods required in each agreement?",
]

QUERY_TRANSFORM_QUERIES = [
    "termination",
    "What happens if I don't pay rent on time?",
    "non-compete",
]

ADAPTIVE_RAG_QUERIES = {
    "Factual": "What is the base rent in the commercial lease?",
    "Comparison": "How do the termination notice periods compare across the service agreement and employment agreement?",
    "Summary": "Summarize the key obligations in the NDA.",
    "Complex": "If Beta Ltd wants to terminate the service agreement early due to a data breach by Acme Corp, what termination and indemnification provisions apply?",
}

CRAG_QUERIES = {
    "In-corpus (should retrieve correctly)": "What is the confidentiality survival period in the NDA?",
    "Out-of-corpus (should trigger fallback)": "What is the trademark registration process for the company logo?",
}

# --- Expanded RAGAS evaluation question bank, spanning the full corpus ------
EVAL_QUESTIONS = [
    "What is the governing law of the service agreement?",
    "What are the termination conditions in the service agreement?",
    "What are the confidentiality obligations in the NDA?",
    "How long do confidentiality obligations survive termination in the NDA?",
    "What is the liability cap in the service agreement?",
    "What is the notice period for the employee to resign?",
    "What is the non-compete duration in the employment agreement?",
    "What is the base rent in the commercial lease?",
    "How much notice does a tenant need to give to renew the lease?",
    "What happens if a tenant defaults on rent in the commercial lease?",
]

# Ground-truth answers for EVAL_QUESTIONS. RAGAS's context_precision and
# context_recall are reference-based metrics — without a `reference` column
# evaluate() raises, so every stock eval question needs one here.
EVAL_REFERENCES = {
    "What is the governing law of the service agreement?":
        "The Service Agreement is governed by the laws of the State of Delaware, "
        "without regard to its conflict of law principles.",
    "What are the termination conditions in the service agreement?":
        "Either party may terminate on thirty (30) days written notice, or "
        "immediately if the other party commits a material breach that remains "
        "uncured for fifteen (15) days after written notice.",
    "What are the confidentiality obligations in the NDA?":
        "The Receiving Party must use Confidential Information solely to evaluate a "
        "potential business relationship, must not disclose it to third parties "
        "without prior written consent, and must protect it with the same degree of "
        "care it uses for its own confidential information, but no less than "
        "reasonable care.",
    "How long do confidentiality obligations survive termination in the NDA?":
        "The confidentiality obligations under Section 3 survive termination for "
        "three (3) years.",
    "What is the liability cap in the service agreement?":
        "Total liability of either party is capped at the fees paid in the preceding "
        "six (6) months, and neither party is liable for indirect, incidental, or "
        "consequential damages.",
    "What is the notice period for the employee to resign?":
        "The Employee must provide two (2) weeks' written notice of resignation.",
    "What is the non-compete duration in the employment agreement?":
        "Twelve (12) months following termination, limited to competing services for "
        "a direct competitor within the United States.",
    "What is the base rent in the commercial lease?":
        "Base rent is $8,500 per month, subject to an annual escalation of three "
        "percent (3%) or the increase in the Consumer Price Index, whichever is greater.",
    "How much notice does a tenant need to give to renew the lease?":
        "At least one hundred eighty (180) days written notice prior to expiration, "
        "to exercise the option for one additional five (5) year term.",
    "What happens if a tenant defaults on rent in the commercial lease?":
        "Failure to pay rent within ten (10) days of the due date is a default; if it "
        "remains uncured for thirty (30) days after written notice, the Landlord may "
        "terminate the Lease and pursue all available remedies.",
}

# --- Clause database for the LangGraph ReAct agent's lookup tool -----------
CLAUSE_DATABASE = {
    "indemnification": (
        "Provider agrees to indemnify and hold harmless Client from any claims "
        "arising from Provider's negligence or willful misconduct. (Service Agreement, Section 5)"
    ),
    "termination": (
        "Either party may terminate the Service Agreement upon thirty (30) days "
        "written notice, or immediately for an uncured material breach after "
        "fifteen (15) days written notice. (Service Agreement, Section 3)"
    ),
    "confidentiality": (
        "The Receiving Party shall not disclose Confidential Information to any "
        "third party without prior written consent; obligations survive "
        "termination for three (3) years. (NDA, Sections 3 and 5)"
    ),
    "liability": (
        "Neither party shall be liable for indirect, incidental, or consequential "
        "damages; total liability is capped at fees paid in the preceding six "
        "(6) months. (Service Agreement, Section 6)"
    ),
    "governing law": (
        "The Service Agreement is governed by Delaware law; the NDA by "
        "California law; the Employment Agreement by New York law; the "
        "Commercial Lease by Illinois law."
    ),
    "non-compete": (
        "For twelve (12) months following termination, the Employee shall not "
        "provide competing services to a direct competitor within the United "
        "States. (Employment Agreement, Section 5.1)"
    ),
    "dispute resolution": (
        "Disputes arising from the Service Agreement are resolved through "
        "binding arbitration in Wilmington, Delaware. (Service Agreement, Section 7)"
    ),
    "rent escalation": (
        "Base rent increases annually by three percent (3%) or the CPI "
        "increase, whichever is greater. (Commercial Lease, Section 3.1)"
    ),
}

# --- Example tasks for the multi-agent supervisor page ----------------------
SUPERVISOR_TASKS = [
    (
        "We're drafting a new vendor services agreement with a data processor "
        "that will handle client PII. Research the relevant clauses and flag "
        "any risk areas before we send it to the client."
    ),
    (
        "A prospective employee negotiating our standard employment agreement "
        "wants to shorten the non-compete from twelve months to three months. "
        "Research our current clause and analyze the risk of agreeing."
    ),
    (
        "Our commercial lease tenant is asking to cap their annual rent "
        "escalation at a flat 2% instead of 3%-or-CPI. Research the current "
        "clause and analyze the financial and precedent risk of accepting."
    ),
]
