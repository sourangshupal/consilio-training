# GDPR & Legal-AI Spotlight — Closing Worksheet

**Day 4 — AI Security & Legal Compliance · Companion to the "GDPR for Engineers & Legal-AI
Spotlight" deck**

This is the **final artifact of the entire 4-day training program**. Like the EU AI Act
worksheet, this is a facilitation document, not a code notebook — GDPR-for-AI and the
Legal-AI Spotlight are discussion and case-study content, best worked through as a group
exercise closing out the program.

---

## Part 1 — Erasure Request Scenarios

For each scenario, work through: is this erasure request straightforward, hard, or somewhere
in between? What would your firm's technical team actually need to DO to comply?

### Scenario A — RAG Index Erasure

> A former client requests, under the right to erasure, that all their personal data be deleted
> from your firm's internal AI research assistant. That assistant is a RAG system that indexed
> the client's case files into a vector store two years ago.

- **Difficulty:** ______________________
- **What needs to happen technically:** ______________________
- **How would you confirm/document compliance?** ______________________

<details>
<summary>Deck's guidance (click to reveal after discussion)</summary>

Straightforward — delete the source documents/chunks from the vector store, and the data is
genuinely gone from what the system can retrieve and surface going forward. Document the
deletion (what was removed, when, confirmation the index no longer returns it) for
accountability purposes.
</details>

### Scenario B — Finetuned Model Erasure

> The same former client's case files were ALSO used, alongside many other clients' anonymized
> matter summaries, to finetune an internal drafting-assistant model six months ago. The client
> requests erasure of their data from this model too.

- **Difficulty:** ______________________
- **What needs to happen technically:** ______________________
- **How would you confirm/document compliance?** ______________________

<details>
<summary>Deck's guidance (click to reveal after discussion)</summary>

Hard — there's no clean "delete this one thing" operation on a model's weights the way there is
for a database row. Options: retrain from scratch without that data (expensive/slow), machine
unlearning techniques (active but still imperfect research area), or — the better answer in
hindsight — this is exactly why the deck recommends keeping personal data in a RAG/retrieval
layer rather than baked into finetuned weights in the first place.
</details>

### Scenario C — Mixed System

> A client-facing legal chatbot uses BOTH a finetuned model (for tone/style) AND a RAG layer
> (for case-specific facts). A data subject requests erasure.

- **Difficulty:** ______________________
- **What needs to happen technically:** ______________________
- **How would you confirm/document compliance?** ______________________

---

## Part 2 — A Compliant AI System Checklist (from the deck)

Walk through this checklist against a real or hypothetical AI tool your firm uses or is
considering. Mark each item Done / In Progress / Not Started, and note who owns it.

| # | Item | Status | Owner |
|---|---|---|---|
| 1 | Lawful basis documented for any data used in training or finetuning | | |
| 2 | RAG preferred over finetuning for content containing personal data | | |
| 3 | Data minimization applied to vector stores and embeddings, not just databases | | |
| 4 | Cross-border transfer implications reviewed for the LLM provider actually used | | |
| 5 | EU AI Act risk tier classified and documented (see the companion worksheet) | | |
| 6 | A defined process exists for erasure AND access requests, covering both RAG index and any finetuned model components separately | | |

---

## Part 3 — Legal-AI Spotlight: Emerging Practice Areas

Discuss, for your own firm:

1. **AI governance advisory** — does your firm currently have anyone positioned to advise
   clients on setting up guardrails, evaluation, and documentation practices? What would it
   take to build that capability?
2. **AI Act risk classification services** — could the classification worksheet exercise from
   this training be turned into a billable client service? What would a first engagement
   look like?
3. **AI vendor contract review** — when your firm signs up for an LLM API provider, a RAG
   platform, or an agent framework, who reviews the data-processing terms and liability
   allocation? Is that review happening today?
4. **AI-related litigation and liability** — as AI systems increasingly cause real-world
   disputes, what technical literacy would your litigation team need to competently handle a
   case involving an AI system's actual behavior?

---

## Part 4 — Closing Reflection

This training began on Day 1 with a simple question: why does AI literacy actually matter for
this audience? Four days and 23 decks later:

- **Day 1** built the foundations (transformers, tokenization, finetuning, prompting) —
  entirely through legal examples.
- **Day 2** covered the tools and frameworks that make finetuning and efficient inference
  practical.
- **Day 3** built RAG systems and agents, from first principles through production-grade
  retrieval and multi-agent architectures.
- **Day 4** closed the loop with security, evaluation, ethics, and legal compliance — the layer
  that makes everything above safe and defensible to actually deploy.

**Closing discussion prompt:** pick ONE thing from across all four days that your firm should
implement first. What's the smallest concrete next step?

______________________________________________________________
______________________________________________________________
______________________________________________________________
