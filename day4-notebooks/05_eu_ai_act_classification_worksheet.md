# EU AI Act — Risk Classification Worksheet

**Day 4 — AI Security & Legal Compliance · Companion to the "EU AI Act & Risk Classification" deck**

This is a facilitation worksheet, not a code notebook — the EU AI Act & Risk Classification
deck is legal/policy content, best worked through as a discussion and classification exercise
rather than forced into a Jupyter notebook.

> **Timeline reminder from the deck:** the original August 2, 2026 high-risk deadline has been
> **provisionally deferred to December 2, 2027** under the "Digital Omnibus" agreement (May 7,
> 2026) — but that deferral is **pending formal adoption**, not settled law. Always verify the
> current status before advising a client, since this is exactly the kind of moving target that
> makes this practice area valuable.

---

## How to Use This Worksheet

For each scenario below:
1. Classify the system into one of the four risk tiers: **Unacceptable / High / Limited / Minimal**
2. Name which specific fact(s) in the scenario drove your classification
3. List the concrete obligations that classification triggers
4. Note anything that would CHANGE the classification if the scenario were slightly different

Work through Scenarios 1-2 as a group first (they mirror the deck's own worked examples), then
use Scenarios 3-6 for independent or small-group practice.

---

## Scenario 1 — Resume Screening Tool (from the deck)

> A law firm is evaluating an AI tool that helps screen resumes and rank candidates for an open
> paralegal position.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

<details>
<summary>Deck's worked answer (click to reveal after discussion)</summary>

High Risk — HR/employment screening and ranking of job applicants is a named Annex III
use case. Triggers documented risk management, human oversight (a human must meaningfully
review rankings, not rubber-stamp them), audit logs, and a conformity assessment.
</details>

---

## Scenario 2 — Client-Facing Chatbot (from the deck)

> A client-facing chatbot on a law firm's website, clearly labeled as AI-powered, that answers
> general questions about the firm's practice areas and office hours.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

<details>
<summary>Deck's worked answer (click to reveal after discussion)</summary>

Limited Risk as described — only needs to disclose that users are interacting with AI.
BUT: if extended to give individualized legal guidance or make any automated decision with
legal/significant effects on a person, it could shift toward High Risk. Classification depends
on actual function, not the "chatbot" label.
</details>

---

## Scenario 3 — Contract Risk-Scoring Tool

> A firm builds an internal tool that scores incoming vendor contracts on a 1-10 "risk" scale,
> which associates use to decide which contracts get expedited partner review versus standard
> review.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

---

## Scenario 4 — Internal Document Summarizer

> An internal tool that summarizes long discovery documents for associates, with no
> decision-making authority — associates read the summary and make their own judgment calls.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

---

## Scenario 5 — Biometric Check-In System

> A firm's office building uses a facial-recognition system to check in visitors and match them
> against a watchlist before granting building access.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

---

## Scenario 6 — Litigation Outcome Predictor (Client-Facing)

> A legal-tech vendor sells a tool to law firms that predicts the likely outcome and settlement
> range of a case based on historical case data, used to advise clients on settlement strategy.

- **Classification:** ______________________
- **Driving fact(s):** ______________________
- **Triggered obligations:** ______________________
- **What would change the classification?** ______________________

---

## Discussion Prompts

1. Which scenario above was the HARDEST to classify confidently, and why? What additional
   information would you need from the client to classify it with certainty?
2. For any system classified as High Risk above, walk through what "documented risk management"
   and "meaningful human oversight" would concretely look like in that firm's actual workflow —
   not just as an abstract requirement.
3. The Digital Omnibus deferral (Aug 2026 → provisionally Dec 2027) is exactly the kind of
   regulatory timeline shift that makes ongoing AI Act advisory valuable. What's your firm's
   process for tracking and communicating a shift like this to clients who received earlier
   advice based on the original deadline?
4. Penalties scale with company size (up to €35M or 7% of global turnover, whichever is higher).
   How does that change the risk calculus for a large multinational client versus a small
   startup client deploying the same AI system?

---

## Quick-Reference: The Practical Classification Checklist (from the deck)

1. What is the system's actual **purpose and use context** — not just its marketing description?
2. Does it appear on, or closely resemble, the **Annex III** list of named high-risk use cases?
3. Does it involve **automated decision-making with legal or otherwise significant effects** on
   individuals?
4. Regardless of tier, what **disclosure obligations** apply — is it clear to users they're
   interacting with an AI system?
5. What is the **current applicable deadline** for any triggered obligations — re-check this,
   don't assume it from an earlier summary.
