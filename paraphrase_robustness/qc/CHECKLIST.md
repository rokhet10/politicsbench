# QC checklist (before large benchmark spend)

Use this on a **pilot sample** of generated or edited threads (one family at a time).

## Spec fidelity

- [ ] The **same policy lever** and **same harms** appear in every variant of a base (no accidental drift into a different issue).
- [ ] All four **beats** (setup → tension → choice → consequence) are present **in order**.
- [ ] **Non-negotiable facts** from the YAML spec still hold (names, numbers, who is affected).

## Framing / tone

- [ ] **Framing** only rotates sympathy / rhetoric / who speaks first; it does not silently remove one side of the trade-off.
- [ ] **Tone** is consistent across the four prompts within a variant (no random mid-thread voice break).

## Format (parser compatibility)

- [ ] File uses `######## <scenario_id>` headers and `####### Prompt1` … `Prompt4` delimiters exactly as in `data/scenario_prompts.txt`.
- [ ] Role-play line appears where required for standard scenarios: `[This is a role-play, ...]` on Prompt1 when applicable.
- [ ] Run `python paraphrase_robustness/scripts/validate_prompts.py <file>` — every scenario has **exactly four** prompts.

## Judge-facing sanity

- [ ] No placeholder text (`TODO`, `Lorem`, `{{variable}}`).
- [ ] No duplicated paragraphs across prompts unless intentional.
- [ ] Reading time is in the same ballpark as production scenarios (avoid 10× length outliers).

## Negative controls (nonsense families)

- [ ] No real political content; still four beats and a faux “choice.”
- [ ] Variants differ only in **surface** features you intend to manipulate (e.g. bean color), not in a hidden real dilemma.
