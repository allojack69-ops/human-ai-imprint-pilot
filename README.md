# Human–AI Imprint Pilot v1.2

A deploy-ready paired experiment for measuring whether a person's habitual AI agent carries a detectable behavioral imprint related to that person.

## Participant flow

1. **Consent**
2. **Human baseline**
   - 18 adaptive easy scenarios
   - 4 medium scenarios
   - 2 hard free-text validation anchors
3. **Personal AI fingerprint**
   - participant copies one prompt into the AI they normally use
   - AI answers 20 short structured scenarios
   - each answer contains `choice + short reason`
4. **Pair stored**
   - human fingerprint
   - personal-AI fingerprint
   - raw human anchors
   - raw AI reasoning
   - reaction times

The participant does not see trait mappings.

## Why v1.2 is stronger than v1.1

The AI phase no longer depends on an external judge to generate the primary fingerprint.
Human and AI choices are scored deterministically from hidden mappings.

Free text is retained as a **secondary** layer for later blind language/reasoning analysis.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ADMIN_KEY="replace-with-a-long-secret"
export DATABASE_PATH="./pilot.db"

uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Persistence

Set `DATABASE_PATH` to a location backed by persistent storage in production.
Do not rely on a temporary container filesystem for real participant data.

## Admin

Live dashboard:

`/admin?key=YOUR_ADMIN_KEY`

JSON summary:

`/admin/summary?key=YOUR_ADMIN_KEY`

Exports:

- `/admin/export.csv?key=...`
- `/admin/export_ai.csv?key=...`
- `/admin/export_fingerprints.json?key=...`

For a more secure deployment, send the key as the `X-Admin-Key` header instead of placing it in the URL.

## Offline analysis

After collecting paired participants:

```bash
python analyze_study.py pilot.db --out analysis_results --permutations 5000
```

Outputs:

- `REPORT.md`
- `pair_matrix.csv`
- `participant_pairs.csv`
- `identification.csv`

Primary tests:

\[
\Delta = E[sim(H_i,A_i)] - E[sim(H_i,A_j)],\quad i\ne j
\]

and blind top-1 identification:

\[
\hat j = \arg\max_j sim(H_i,A_j)
\]

The script also computes a bootstrap confidence interval for the matched-vs-mismatched gap and a permutation p-value for identification accuracy.

## Synthetic pipeline test

Never mix synthetic data with real study data.

```bash
cp pilot.db synthetic_test.db
python generate_synthetic.py synthetic_test.db
python analyze_study.py synthetic_test.db --out synthetic_results
```

## Hidden primary dimensions

The current deterministic layer samples dimensions including:

- state dependence
- information value
- resilience / redundancy
- systems and relation orientation
- control
- goal definition
- uncertainty handling
- tail-risk sensitivity
- diversity
- causal discipline
- reversibility
- option value
- model synthesis
- bidirectional audit
- failure localization
- cheap-probe preference
- performance preference

These are **experimental latent features**, not clinical or psychometric diagnoses.

## Methodological gates before any strong claim

Do not claim a human-specific AI imprint merely because matched pairs look similar.

Require:

1. matched > mismatched similarity;
2. bootstrap CI supporting a positive gap;
3. blind identification above chance;
4. permutation support;
5. replication on held-out participants;
6. control for base model/model family;
7. later style-normalized free-text analysis;
8. repeat-session reliability.

## Next milestone: v1.2

Add:
- standardized control-AI responses for every participant;
- repeated probe subset for test-retest reliability;
- model-family stratification;
- held-out prediction set;
- automatic power/sample-size simulation.


## v1.2 contamination control

For the personal-AI phase, participants should:

1. use their **usual personal AI account**;
2. open a **new normal chat**;
3. avoid Temporary/Incognito mode if it disables the personalization being studied;
4. never paste their human-test answers into the AI chat;
5. run exactly the provided standardized prompt.

The study also records low-burden metadata:
- model/service name;
- memory on/off/unknown;
- custom instructions on/off/unknown;
- rough duration of use.

## Model-family confound control

A raw result such as:

`sim(H_i, A_i) > sim(H_i, A_j)`

can be misleading when `A_i` and `A_j` are different base model families.

v1.2 therefore also computes:

`matched human↔AI similarity - mismatched similarity within the same model family`

and performs top-1 identification restricted to same-family competitors plus a family-stratified permutation test.

This is a required gate before attributing the signal to a human-specific imprint.
