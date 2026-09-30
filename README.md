# Samarthan — SIH26094 Functional POC

Samarthan is an engineering proof-of-concept for AI-assisted continuous distress triage and human-led support for PoA victims/complainants.

## Demo workflow

1. **Victim Portal** — calm, minimal check-in experience for text/voice/silent-duress simulation.
2. **AI triage** — lightweight TF-IDF + multinomial logistic regression plus transparent sanity calibration, personal-baseline deviation and channel signals.
3. **Counsellor Console** — priority queue, longitudinal context, explainability and recommended next step.
4. **Human review** — counsellor records an intervention and can tick **Mark this check-in as reviewed**.
5. **Victim notification** — the victim-facing acknowledgement updates to **Counsellor reviewed** and shows the follow-up status.
6. **Audit** — review/action events are recorded.

## Run locally

```text
py -3.14 -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Demo routes

- `/` — role selection
- `/victim` — victim portal
- `/counsellor` — counsellor console
- `/queue` — priority queue
- `/validation` — engineering validation
- `/integrations` — mock government/system adapters
- `/audit` — audit trail
- `/api/health` — health endpoint

## Responsible AI boundaries

This prototype does **not** diagnose mental illness, predict suicide, determine credibility, detect deception, compare victims against one another, or make autonomous intervention decisions.

**AI = triage support. Human = final decision.**

The text model and calibration layer are engineering/demo components trained/evaluated on synthetic examples. Results are **not clinical validation** and must not be used for real-world care or protection decisions.

## Integrations

NHAA 14566, e-Courts, PFMS and Bhashini are represented as mock/adapter interfaces only. No live government data or credentials are included.


## v0.8 enhancement layer — inspired by PS26094 peer-project patterns

This version keeps the original Samarthan AI triage core and adds four bounded engineering layers:

1. **Early-warning trajectory:** recent time-stamped scores produce a bounded distress velocity and 14-day decision-support projection. Projection is withheld when valid timing evidence is insufficient.
2. **Closed-loop interventions:** recommendations can move through `recommended → approved → dispatched → completed` or `dismissed`, with human notes and audit events.
3. **Consent-gated multi-channel follow-up:** Web/App/SMS/IVRS are represented as replaceable channels. A failed preferred channel moves to the next configured fallback; exhausted channels create a contact-continuity review rather than treating non-response as non-compliance.
4. **Tier-2 aggregate monitoring:** the `/admin` view shows aggregate case/check-in/intervention metrics and district cells. Counts below five are suppressed to reduce re-identification risk. No individual narrative or case ID is exposed by this view.

### Important prototype boundaries

- SMS/IVRS/Web/App delivery is simulated; no live telecom or government API is claimed.
- The trajectory engine is decision-support mathematics, not a clinical prediction model.
- The text classifier remains a dependency-light TF-IDF + multinomial Logistic Regression trained on synthetic engineering examples.
- Existing role separation remains a workflow demonstration, not production authentication. Production deployment requires server-side identity, authorisation, consent, encryption, retention and institutional governance.

### Suggested judge demo

`Victim check-in → AI signal → rising trajectory → counsellor queue → explainability → accept/dispatch intervention → schedule follow-up → simulate channel failure/fallback → victim responds → audit → aggregate admin view`.

This sequence demonstrates the PS26094 loop without requiring external services.
