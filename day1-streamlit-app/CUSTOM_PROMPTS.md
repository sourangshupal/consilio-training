# Custom Prompt Library — Day 1 Streamlit App

Copy-paste ready inputs for every free-text field in the app (`custom prompt`,
`custom clause`, `...or paste your own`, etc). Organized by page, in the order
the fields appear. Each preset the field would otherwise default to is noted
so you know what you're overriding.

---

## 01 — Transformers 101 (`pages/01_transformers_101.py`)

**Field:** *"...or type your own sentence (overrides the dropdown)"* — `st.text_input`
Feeds the live attention heatmap. Default dropdown draws from `ATTENTION_SENTENCES`
(Indemnification / Termination / Confidentiality / Arbitration / Force Majeure /
Liability Cap).

Paste one of these to see how attention heads distribute across a denser or more
ambiguous clause than the built-in six:

```
The Purchaser shall have the right, but not the obligation, to conduct a due diligence review of the Target's financial records within sixty days of the Effective Date, provided that such review shall not unreasonably interfere with the Target's ordinary course of business.
```

```
Notwithstanding anything to the contrary herein, no amendment, modification, or waiver of any provision of this Agreement shall be effective unless set forth in a writing signed by both parties.
```

```
The Licensee shall not sublicense, assign, or otherwise transfer any rights granted under this Agreement without the prior written consent of the Licensor, which consent shall not be unreasonably withheld.
```

---

## 02 — Tokenization (`pages/02_tokenization.py`)

**Field:** *"Paste a clause"* — `st.text_area` (line 77)
Feeds the tokenizer shootout (compares how different tokenizers split legal
language — good for showing subword fragmentation on Latin/legal terms).

```
The parties acknowledge and agree that, notwithstanding the doctrine of res judicata, any dispute not raised in the initial arbitration proceeding shall not be barred by collateral estoppel, and each party reserves the right to seek a writ of certiorari should the arbitral award be challenged on jurisdictional grounds.
```

```
Pursuant to the indemnification provisions set forth in Section 8, the Indemnitor shall hold harmless the Indemnitee from and against any and all claims, liabilities, and quantum meruit demands arising in camera or otherwise, subject to the caveat emptor principles governing this transaction.
```

```
The parties stipulate that voir dire shall proceed in accordance with the jurisdiction's habeas corpus statutes, and any subpoena duces tecum issued hereunder shall be governed by the mens rea and actus reus standards applicable to the underlying prima facie case.
```

---

## 03 — KV Cache Speed Benchmark (`pages/03_kv_cache.py`)

**Field:** *"Prompt"* — `st.text_area` (line 38, prefilled with `DEFAULT_PROMPT`, an MSA excerpt)
Feeds a local DistilGPT-2 generation benchmark (cached vs. uncached). Longer
prompts show a bigger cache speedup — use these to demonstrate that scaling
effect more dramatically than the short default.

```
This Software License and Services Agreement ("Agreement") is entered into by and between Acme Corporation, a Delaware corporation ("Licensor"), and the undersigned licensee ("Licensee"). Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Licensor's proprietary software solely for Licensee's internal business purposes, subject to the terms and conditions set forth herein. Licensee shall not reverse engineer, decompile, or disassemble the software, nor shall Licensee sublicense, resell, or distribute the software to any third party without the prior written consent of Licensor. Licensor shall provide standard technical support during normal business hours and shall use commercially reasonable efforts to correct any material defects reported by Licensee within a reasonable time. This Agreement shall remain in effect for an initial term of twelve months and shall automatically renew for successive twelve-month terms unless either party provides written notice of non-renewal at least thirty days prior to the end of the then-current term.
```

```
Confidentiality Agreement: The Receiving Party acknowledges that it will have access to certain confidential and proprietary information of the Disclosing Party, including but not limited to trade secrets, business plans, financial data, customer lists, and technical specifications. The Receiving Party agrees to hold all such Confidential Information in strict confidence and shall not disclose it to any third party without the prior written consent of the Disclosing Party, except as required by law or court order, in which case the Receiving Party shall provide prompt written notice to the Disclosing Party to allow it to seek a protective order. This obligation of confidentiality shall survive the termination of this Agreement for a period of five years, except with respect to trade secrets, which shall remain confidential for as long as they retain trade secret status under applicable law.
```

---

## 04 — Model Landscape — Legal Angle (`pages/04_model_landscape.py`)

### Part 1 — LegalBERT zero-shot clause classification

**Field:** *"...or paste your own clause (overrides the dropdown)"* — `st.text_area` (line 41)
Default dropdown: `CLASSIFY_TARGET_CLAUSES` (Termination / Confidentiality /
Liability Cap). Scored against the 10-label `CLAUSE_TAXONOMY`. Paste a clause
whose category is genuinely ambiguous to show where cosine-similarity
classification breaks down:

```
This Agreement, and any rights or obligations hereunder, may not be assigned by either party without the prior written consent of the other party, except that either party may assign this Agreement without consent in connection with a merger, acquisition, or sale of substantially all of its assets.
```

```
If any provision of this Agreement is held to be invalid or unenforceable, such provision shall be struck and the remaining provisions shall be enforced to the fullest extent permitted by law, and the parties shall negotiate in good faith to replace the invalid provision with a valid one that most closely approximates the original intent.
```

```
Any controversy or claim arising out of or relating to this contract shall first be submitted to non-binding mediation, and if unresolved within sixty days, shall thereafter be resolved exclusively through binding arbitration administered by JAMS in accordance with its Comprehensive Arbitration Rules.
```

### Part 2 — Qwen2.5-0.5B-Instruct free generation

**Field:** *"...or write your own prompt"* — `st.text_area` (line 68)
Default dropdown: `QWEN_PROMPT_PRESETS` (Attorney Advertising Disclaimer /
Plain-Language Clause Summary / Client Email Draft). Try these to stress-test
a 0.5B model on realistic law-firm drafting asks:

```
Draft a two-sentence out-of-office auto-reply for a litigation associate who is unreachable for one week due to trial.
```

```
Rewrite this clause in plain, client-friendly language: "In no event shall either party's aggregate liability exceed the total fees paid under this Agreement in the twelve months preceding the claim."
```

```
Write a short LinkedIn post (under 50 words) announcing that the firm's corporate practice group added a new partner specializing in cross-border M&A.
```

```
Summarize the following in one sentence for a client update email: The court granted our motion to dismiss two of the plaintiff's five claims but allowed the negligence and breach of contract claims to proceed to discovery.
```

---

## 05 — Legal Prompt Engineering Playground (`pages/05_prompt_engineering.py`)

### Technique 1 — Zero-shot vs. few-shot clause classification

**Field:** *"Clause to classify"* — `st.text_area`, key `zs_clause` (default: Confidentiality clause)

```
This Agreement and all rights and licenses granted hereunder shall automatically terminate if the Licensee fails to cure a material breach within fifteen days of receiving written notice thereof from the Licensor.
```

```
Each party represents and warrants that it has full corporate power and authority to enter into this Agreement and that the execution and performance of this Agreement will not violate any other agreement to which it is a party.
```

### Technique 5 — JSON-mode clause extraction

**Field:** *"Clause to extract metadata from"* — `st.text_area`, key `json_clause` (default: Indemnification clause)
Extracts `clause_type`, `obligated_party`, `beneficiary_party`, `conditions`, `deadline_days`.

```
The Contractor shall indemnify, defend, and hold harmless the Owner from any and all third-party claims arising out of the Contractor's negligent acts or omissions in the performance of the work, provided that the Owner gives the Contractor written notice of any such claim within thirty days of receiving it.
```

```
The Tenant shall indemnify the Landlord against any loss or damage arising from the Tenant's breach of the building's fire safety code, unless such breach resulted from a defect in the premises that existed prior to the Tenant's occupancy and was not disclosed to the Tenant.
```

### Technique 5 — XML-tag reasoning/answer extraction

**Field:** *"Dispute description"* — `st.text_area`, key `xml_fact`
(default: a CPI-indexed rent escalation dispute)

```
A supply agreement sets a fixed delivery schedule of the first Monday of each month. The supplier missed three consecutive months, citing a force majeure clause that references "acts of God, war, and government action" — but the actual cause was the supplier's own factory understaffing.
```

```
An employment contract includes a one-year non-compete clause covering "any business in the software industry" with no geographic limitation. The former employee took a job at a company in a different state that does not compete in the same product category.
```

### Technique 6 — Prompt chaining (3-stage pipeline)

**Field:** *"Case summary"* — `st.text_area`, key `chain_case` (default: `SLIP_AND_FALL_SUMMARY`)
Runs through: extract facts → draft Statement of Facts → validate against source.

```
Case: Ortiz v. Meridian Freight Co. On September 14, plaintiff Carlos Ortiz was rear-ended by a Meridian Freight delivery truck while stopped at a red light. Ortiz suffered a herniated disc and has been unable to return to his job as a warehouse loader. The Meridian driver's dashcam footage shows the driver looking down at a handheld device for approximately four seconds immediately before impact. Meridian's internal logs show the driver had been on shift for eleven hours, two hours past the company's stated safety limit.
```

```
Case: Chen v. Brightline Realty Partners. Brightline listed a commercial property as "zoned for retail use" in its offering memorandum. Buyer Wei Chen purchased the property for $2.4 million relying on that representation. After closing, Chen discovered the property was zoned for light industrial use only, and retail use required a variance that the city denied. Brightline's listing agent admitted in a deposition that she never confirmed the zoning designation with the city before publishing the memorandum.
```

---

## Notes

- Fields with a dropdown + override (`04_model_landscape.py`, `01_transformers_101.py`)
  only use your pasted text if it's non-empty — clear the box to fall back to
  the preset.
- All "Run live" buttons need an API key set in the sidebar; static/offline
  pages (Transformers 101, Tokenization, KV Cache, LegalBERT/Qwen in Model
  Landscape) work without one.
- These are illustrative fact patterns, not real client matters — safe to
  paste into any provider (OpenAI / Gemini / Groq) during testing.
