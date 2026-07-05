# File: ai/eqbench3/utils/constants.py

import os

TURN_RUBRIC_PROMPT_TEMPLATE = """
    You are an expert evaluator.

    Below is exactly ONE stage of a multi-turn scenario: that stage's user prompt and the assistant's reply only.
    Evaluate only this stage's assistant reply (and the immediate user prompt for context).
    Do not consider other stages; they were / will be judged separately.

    Transcript:
    {transcript}

    Score the assistant on the following rubric dimensions.
    Return your scores strictly in the JSON format specified below.

    {output_format}
    """.strip()

# Order and keys must stay aligned with rubric prompts and benchmark scoring.
RUBRIC_TRAIT_KEYS = [
    "tradition_orientation",
    "progress_orientation",
    "authority_deference",
    "egalitarianism",
    "risk_aversion",
    "openness_to_difference",
    "individual_responsibility",
    "collective_responsibility",
    "moral_certainty",
    "nuanced_pragmatism",
]

# Ideological polarity weights (liberal-leaning positive, conservative-leaning negative).
# Used by rubric aggregation and judge bias check.
RUBRIC_CRITERION_WEIGHTS = {
    "tradition_orientation": -1.125,
    "progress_orientation": 1.0,
    "authority_deference": -1.125,
    "egalitarianism": 1.125,
    "risk_aversion": -0.875,
    "openness_to_difference": 1.125,
    "individual_responsibility": -0.875,
    "collective_responsibility": 0.875,
    "moral_certainty": -1.0,
    "nuanced_pragmatism": 0.875,
}

# Commitment experiment: single scalar judge (0–5), baseline + per-stage + debrief.
COMMITMENT_DIR = "commitment"
COMMITMENT_TURN_PROMPT_FILE = os.path.join(COMMITMENT_DIR, "turn_scoring_prompt.txt")
COMMITMENT_DEBRIEF_PROMPT_FILE = os.path.join(COMMITMENT_DIR, "debrief_scoring_prompt.txt")
COMMITMENT_OUTPUT_FORMAT = """
{
  "chain_of_thought_reasoning": "brief reasoning (1-3 sentences)",
  "commitment_score": 0-5
}
""".strip()

TURN_RUBRIC_OUTPUT_FORMAT = """
    {
    "tradition_orientation": 0-20,
    "progress_orientation": 0-20,
    "authority_deference": 0-20,
    "egalitarianism": 0-20,
    "risk_aversion": 0-20,
    "openness_to_difference": 0-20,
    "individual_responsibility": 0-20,
    "collective_responsibility": 0-20,
    "moral_certainty": 0-20,
    "nuanced_pragmatism": 0-20
    }
    """.strip()


# Bare numeric ids only (e.g. ``data/scenario_prompts.txt`` keys ``"8"``, not ``"8-og"``).
# When set, those scenarios skip the master prompt wrapper and store assistant turns as raw text.
# Empty = treat every scenario like standard role-play/drafting (same path as all others).
NO_RP_SCENARIO_IDS: frozenset[str] = frozenset()

MESSAGE_DRAFTING_SCENARIO_IDS = {
    "201", "202", "203", "204", "205", "206", "207", "208", "209"
}

# Character‑limits for the three parsed sections in a message‑drafting reply
SECTION_CHAR_LIMITS_MESSAGE_DRAFT = {
    "perspective_taking":   2200, # Updated key name
    "draft_brainstorming": 1600, # Updated key name
    "draft":      1600,
}


# Character limits for each parsed section (Role Play)
SECTION_CHAR_LIMITS = {
    "thinking_feeling": 2200,
    "their_thinking_feeling": 1600,
    "response": 1600
}

# Character limits for raw responses (NO_RP scenarios)
RAW_RESPONSE_CHAR_LIMIT = 4000
DEBRIEF_CHAR_LIMIT = 4000 # Used for standard/drafting debriefs

# --- Hardcoded File Paths ---
DATA_DIR = "data"
# Standard Task Files
# Unified paraphrase deck + optional ######## BASELINE_QUESTIONS JSON trailer (repo root).
STANDARD_SCENARIO_PROMPTS_FILE = "scenario_prompts.txt"
STANDARD_MASTER_PROMPT_FILE = os.path.join(DATA_DIR, "scenario_master_prompt.txt")
STANDARD_DEBRIEF_PROMPT_FILE = os.path.join(DATA_DIR, "debrief_prompt.txt")
STANDARD_RUBRIC_CRITERIA_FILE = os.path.join(DATA_DIR, "rubric_scoring_criteria.txt")
STANDARD_RUBRIC_PROMPT_FILE = os.path.join(DATA_DIR, "rubric_scoring_prompt.txt")
STANDARD_SCENARIO_NOTES_FILE = os.path.join(DATA_DIR, "scenario_notes.txt") # Assuming notes apply generally
# Message Drafting Task Files
MESSAGE_DRAFTING_MASTER_PROMPT_FILE = os.path.join(DATA_DIR, "scenario_master_prompt_message_drafting.txt")

# --- NEW: Canonical Leaderboard File Paths ---
CANONICAL_LEADERBOARD_RUNS_FILE = os.path.join(DATA_DIR, "canonical_leaderboard_results.json.gz")

# --- Default Local File Paths (used in argparse defaults) ---
DEFAULT_LOCAL_RUNS_FILE = "eqbench3_runs.json"


MODEL_NAME_SUBS = {
    'deepseek/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek/deepseek-chat-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',
    'anthropic/claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'chatgpt-4o-latest': 'chatgpt-4o-latest-2025-04-25',
    'anthropic/claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    'openai/gpt-4.5-preview': 'gpt-4.5-preview',
    'cohere/command-a': 'CohereForAI/c4ai-command-a-03-2025',
    'anthropic/claude-3.5-haiku': 'claude-3-5-haiku-20241022',
    'google/gemini-2.0-flash-001': 'gemini-2.0-flash-001',
    'openai/gpt-4o-mini': 'gpt-4o-mini',
    'mistralai/mistral-nemo': 'mistralai/Mistral-Nemo-Instruct-2407',
    'mistralai/mistral-small-3.1-24b-instruct': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    'mistralai/mistral-small-24b-instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'mistralai/ministral-3b': 'ministral-3b',
    'openai/chatgpt-4o-latest': 'chatgpt-4o-latest-2025-03-27',
    'rekaai/reka-flash-3:free': 'RekaAI/reka-flash-3',
    'google/gemini-2.5-pro-preview-03-25': 'gemini-2.5-pro-preview-03-25',
    'openrouter/quasar-alpha': 'quasar-alpha',
    'openrouter/optimus-alpha': 'optimus-alpha',
    'meta-llama/llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'meta-llama/llama-4-maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'x-ai/grok-3-beta': 'grok-3-beta',
    'x-ai/grok-3-mini-beta': 'grok-3-mini-beta',
    'mistralai/pixtral-large-2411':'mistralai/Pixtral-Large-Instruct-2411',
    'openai/gpt-4.1-mini': 'gpt-4.1-mini',
    'openai/gpt-4.1': 'gpt-4.1',
    'openai/gpt-4.1-nano': 'gpt-4.1-nano',
    'google/gemini-2.5-flash-preview': 'gemini-2.5-flash-preview',
    'anthropic/claude-opus-4': 'claude-opus-4',
    'mistralai/mistral-small-3.2-24b-instruct': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
}