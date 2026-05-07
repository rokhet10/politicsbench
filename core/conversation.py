
# File: ai/eqbench3/core/conversation.py

# core/conversation.py

import time
import logging
import re
import json # For parsing rubric scores
from typing import Dict, Any, List, Optional, Tuple
import queue
from concurrent.futures import ThreadPoolExecutor

from core.judge_suite import aggregate_rubric_scores
from utils.constants import (
    NO_RP_SCENARIO_IDS,
    MESSAGE_DRAFTING_SCENARIO_IDS,
    ANALYSIS_SCENARIO_IDS, # Added
    SECTION_CHAR_LIMITS,
    SECTION_CHAR_LIMITS_MESSAGE_DRAFT,
    RAW_RESPONSE_CHAR_LIMIT,
    DEBRIEF_CHAR_LIMIT,
    ANALYSIS_RESPONSE_CHAR_LIMIT,
    TURN_RUBRIC_PROMPT_TEMPLATE,
    TURN_RUBRIC_OUTPUT_FORMAT,
)
from utils.utils import robust_json_loads

class ScenarioTask:
    """
    Represents a single multi-turn scenario simulation task for eqbench3.
    Handles running the scenario prompts sequentially, an optional final debrief prompt
    (skipped for analysis tasks), per-stage rubric scoring during the scenario, and an
    optional final rubric step (debrief-only for standard tasks; last stage for analysis).
    Includes iteration_index.
    Manages its own state but relies on external orchestrator (benchmark.py) for API calls.
    Stores the logical model name internally.
    """

    BASELINE_USER_PREFIX = (
        "Answer in first person as yourself. Aim for roughly 1000 words in total—"
        "similar depth and length to your replies in the full scenario format "
        "(not a short paragraph summary).\n\n"
    )

    def __init__(
        self,
        scenario_id: str,
        prompts: List[str],
        debrief_prompt: Optional[str], # Can be None for analysis tasks
        iteration_index: int,
        test_model: str, # This is the logical model_name passed from benchmark.py
        master_prompt_template: str = None,
        baseline_prompt: Optional[str] = None,
    ):
        self.scenario_id = scenario_id
        self.prompts = prompts
        self.debrief_prompt = debrief_prompt # Store None if analysis task
        self.iteration_index = iteration_index
        self.model_name = test_model # Store the logical name internally
        self.master_prompt_template = master_prompt_template
        self.baseline_prompt = (baseline_prompt.strip() if baseline_prompt else None) or None

        # Status Lifecycle:
        # Standard/Drafting: initialized -> running_scenario -> scenario_completed -> running_debrief -> completed -> [running_rubric_scoring -> rubric_scored]
        # Analysis:          initialized -> running_scenario -> scenario_completed -> [running_rubric_scoring -> rubric_scored]
        # Any step can transition to 'error'.
        self.status = "initialized"
        self.start_time = None
        self.end_time = None # Marks end of the *entire* task processing (incl. rubric)
        self.error = None # General error message if any step fails

        # Scenario Phase Data
        self.conversation_history: List[Dict[str, str]] = []
        self.parsed_responses: List[Dict[str, str]] = [] # Stores parsed sections or just {"raw": ...} for NO_RP/ANALYSIS
        self.scenario_run_error: Optional[str] = None

        # Debrief Phase Data (Only for non-analysis tasks)
        self.debrief_response: Optional[str] = None
        self.debrief_run_error: Optional[str] = None

        # Rubric Scoring Phase Data
        self.rubric_scores: Optional[Dict[str, float]] = None
        self.raw_rubric_judge_text: Optional[str] = None
        self.rubric_scores_by_judge: Optional[List[Dict[str, float]]] = None
        self.raw_rubric_judge_text_by_judge: Optional[List[str]] = None
        self.rubric_run_error: Optional[str] = None

        self.turn_rubric_scores: List[Optional[Dict[str, float]]] = []
        self.turn_raw_rubric_judge_text: List[Optional[str]] = []
        self.turn_rubric_scores_by_judge: List[Optional[List[Dict[str, float]]]] = []
        self.turn_raw_rubric_judge_text_by_judge: List[Optional[List[str]]] = []

        # Optional blanket baseline (same topic, no scenario framing); not part of staged transcript.
        self.baseline_conversation_history: List[Dict[str, str]] = []
        self.baseline_rubric_scores: Optional[Dict[str, float]] = None
        self.baseline_raw_rubric_judge_text: Optional[str] = None
        self.baseline_rubric_scores_by_judge: Optional[List[Dict[str, float]]] = None
        self.baseline_raw_rubric_judge_text_by_judge: Optional[List[str]] = None
        self.baseline_parsed_response: Optional[Dict[str, str]] = None

        # Commitment judge (0–5), separate from trait rubrics when both run
        self.baseline_commitment_scores: Optional[Dict[str, float]] = None
        self.baseline_raw_commitment_judge_text: Optional[str] = None
        self.baseline_commitment_scores_by_judge: Optional[List[Dict[str, float]]] = None
        self.baseline_raw_commitment_judge_text_by_judge: Optional[List[str]] = None
        self.turn_commitment_scores: List[Optional[Dict[str, float]]] = []
        self.turn_raw_commitment_judge_text: List[Optional[str]] = []
        self.turn_commitment_scores_by_judge: List[Optional[List[Dict[str, float]]]] = []
        self.turn_raw_commitment_judge_text_by_judge: List[Optional[List[str]]] = []
        self.debrief_commitment_scores: Optional[Dict[str, float]] = None
        self.raw_debrief_commitment_judge_text: Optional[str] = None
        self.debrief_commitment_scores_by_judge: Optional[List[Dict[str, float]]] = None
        self.raw_debrief_commitment_judge_text_by_judge: Optional[List[str]] = None

    def _save_progress(self, save_queue: Optional[queue.Queue], run_key: Optional[str]):
        """Helper to put the current task state onto the save queue."""
        if save_queue and run_key:
            try:
                task_data = self.to_dict()
                save_queue.put((run_key, self.iteration_index, self.scenario_id, task_data))
                logging.debug(f"Task {self.scenario_id} (Iter {self.iteration_index}) data queued for saving (Status: {self.status}).")
            except Exception as e:
                 logging.error(f"Failed to queue save progress for task {self.scenario_id} (Iter {self.iteration_index}): {e}", exc_info=True)
        elif not save_queue:
             logging.warning("Save queue is None, cannot save progress.")
        elif not run_key:
             logging.warning("Run key is None, cannot save progress.")


    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Extract the structured sections from a model reply during scenario turns.
        Only applicable for standard role-play and message drafting scenarios.
        """
        # Patterns for different scenario types that require parsing
        # core/elo.py  (or wherever you define _parse_response)
        draft_patterns = {
            "perspective_taking":   r"#\s*Perspective[- ]taking\s*\n([\s\S]*?)(?=#|\Z)",
            "draft_brainstorming":  r"#\s*Draft brainstorming\s*\n([\s\S]*?)(?=#|\Z)",
            "draft":                r"#\s*Draft\s*\n([\s\S]*?)(?=--|\Z)",
        }

        # Accept straight (') **or** curly (’ U+2019) apostrophes
        rp_patterns = {
            "thinking_feeling":
                r"#\s*I[’']m thinking\s*\n([\s\S]*?)(?=#|\Z)",
            "their_thinking_feeling":
                r"#\s*They[’']re thinking & feeling\s*\n([\s\S]*?)(?=#|\Z)",
            "response":
                r"#\s*My response\s*\n([\s\S]*?)(?=--|\Z)",
        }


        # Determine which patterns to use
        if self.scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS:
            patterns = draft_patterns
        elif self.scenario_id not in NO_RP_SCENARIO_IDS and self.scenario_id not in ANALYSIS_SCENARIO_IDS:
            patterns = rp_patterns
        else:
            # Should not be called for NO_RP or ANALYSIS, but return raw if it is
            logging.warning(f"_parse_response called unexpectedly for scenario {self.scenario_id}. Returning raw.")
            return {"raw": response}

        parsed = {k: "" for k in patterns}
        parsed["raw"] = response # Always include raw response

        for key, pat in patterns.items():
            m = re.search(pat, response, flags=re.IGNORECASE | re.DOTALL) # Added DOTALL
            if m:
                parsed[key] = m.group(1).strip()

        return parsed

    @staticmethod
    def _parse_rubric_scores(judge_response_text: str) -> Optional[Dict[str, float]]:
        """
        Parses the JSON response from the rubric scoring judge.
        Returns a dictionary of scores or None if parsing fails.
        Looks for content between matching { and } characters.
        """
        try:
            # Find the first '{' and last '}' to extract JSON content
            #start_idx = judge_response_text.find('{')
            #end_idx = judge_response_text.rfind('}')

            #if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            #    logging.warning(f"No valid JSON object found. Raw text: {judge_response_text}...")
            #    return None

            #json_str = judge_response_text[start_idx:end_idx+1].strip()
            parsed_data = robust_json_loads(judge_response_text)

            if not isinstance(parsed_data, dict):
                logging.warning(f"Rubric score parsing failed: Parsed data is not a dictionary. Raw text: {judge_response_text}...")
                return None
            
            if not parsed_data:
                logging.warning(f"Rubric score parsing yielded no numeric scores. Raw text: {judge_response_text}\n\nParsed: {json.dumps(parsed_data)}")
                return None

            scores = {}
            for key, value in parsed_data.items():
                if isinstance(value, (int, float)):
                    scores[key] = float(value)
                elif key != "chain_of_thought_reasoning": # Allow reasoning string
                    # Try to extract number if string contains one (e.g., "15/20")
                    num_match = re.search(r'\b(\d+(\.\d+)?)\b', str(value))
                    if num_match:
                        try:
                            scores[key] = float(num_match.group(1))
                            logging.debug(f"Extracted numeric score {scores[key]} from string '{value}' for key '{key}'.")
                        except ValueError:
                             logging.debug(f"Rubric score for '{key}' is non-numeric ('{value}'). Skipping.")
                    else:
                        logging.debug(f"Rubric score for '{key}' is non-numeric ('{value}'). Skipping.")

            if not scores:
                logging.warning(f"Rubric score parsing yielded no numeric scores. Raw text: {judge_response_text}\n\nParsed: {json.dumps(parsed_data)}")
                return None

            if "commitment_score" in scores:
                scores["commitment_score"] = max(
                    0.0, min(5.0, float(scores["commitment_score"]))
                )

            return scores

        except json.JSONDecodeError as e:
            logging.warning(f"Rubric score JSON parsing failed: {e}. Raw text: {judge_response_text}...")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during rubric score parsing: {e}", exc_info=True)
            return None

    def _format_assistant_content_for_rubric(
        self, assistant_response_index: int, truncate_for_rubric: bool
    ) -> str:
        """Format one assistant turn for rubric prompts (parsed sections or raw)."""
        is_analysis = self.scenario_id in ANALYSIS_SCENARIO_IDS
        is_no_rp = self.scenario_id in NO_RP_SCENARIO_IDS
        is_drafting = self.scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS

        raw_content = ""
        if 2 * assistant_response_index + 1 < len(self.conversation_history):
            raw_content = self.conversation_history[2 * assistant_response_index + 1].get(
                "content", ""
            )

        if truncate_for_rubric:
            if is_no_rp:
                return ScenarioTask._truncate_text(raw_content, RAW_RESPONSE_CHAR_LIMIT)
            if is_analysis:
                return ScenarioTask._truncate_text(
                    raw_content, ANALYSIS_RESPONSE_CHAR_LIMIT
                )
            try:
                if assistant_response_index >= len(self.parsed_responses):
                    raise IndexError(
                        f"Assistant response index {assistant_response_index} out of bounds "
                        f"for parsed_responses (len {len(self.parsed_responses)})"
                    )
                parsed = self.parsed_responses[assistant_response_index]
                if not isinstance(parsed, dict):
                    logging.warning(
                        f"Parsed response at index {assistant_response_index} is not a dict "
                        f"for task {self.scenario_id}. Using raw truncated."
                    )
                    return ScenarioTask._truncate_text(raw_content, RAW_RESPONSE_CHAR_LIMIT)
                limits = (
                    SECTION_CHAR_LIMITS_MESSAGE_DRAFT
                    if is_drafting
                    else SECTION_CHAR_LIMITS
                )
                truncated_sections = ScenarioTask._truncate_parsed_sections(parsed, limits)
                return ScenarioTask._format_parsed_response(truncated_sections)
            except IndexError as e:
                logging.error(
                    f"{e} accessing parsed_responses for task {self.scenario_id} "
                    f"(Iter {self.iteration_index}). Using raw truncated content."
                )
                return ScenarioTask._truncate_text(raw_content, RAW_RESPONSE_CHAR_LIMIT)
            except Exception as e:
                logging.error(
                    f"Error processing/truncating parsed response {assistant_response_index} "
                    f"for task {self.scenario_id} (Iter {self.iteration_index}): {e}. "
                    "Using raw truncated content.",
                    exc_info=True,
                )
                return ScenarioTask._truncate_text(raw_content, RAW_RESPONSE_CHAR_LIMIT)

        if is_no_rp or is_analysis:
            return raw_content
        try:
            if assistant_response_index >= len(self.parsed_responses):
                raise IndexError(
                    f"Assistant response index {assistant_response_index} out of bounds "
                    f"for parsed_responses (len {len(self.parsed_responses)})"
                )
            parsed = self.parsed_responses[assistant_response_index]
            if not isinstance(parsed, dict):
                logging.warning(
                    f"Parsed response at index {assistant_response_index} is not a dict "
                    f"for task {self.scenario_id}. Using raw."
                )
                return raw_content
            return ScenarioTask._format_parsed_response(parsed)
        except IndexError as e:
            logging.error(
                f"{e} accessing parsed_responses for task {self.scenario_id} "
                f"(Iter {self.iteration_index}). Using raw content.",
                exc_info=True,
            )
            return raw_content
        except Exception as e:
            logging.error(
                f"Error formatting parsed response {assistant_response_index} "
                f"for task {self.scenario_id} (Iter {self.iteration_index}): {e}. "
                "Using raw content.",
                exc_info=True,
            )
            return raw_content

    def build_single_stage_rubric_transcript(
        self, stage_index: int, truncate_for_rubric: bool
    ) -> Optional[str]:
        """
        User prompt + assistant reply for one stage only (0-based stage index).
        Used for per-turn rubric judging and for analysis final rubric (last stage only).
        """
        need = 2 * (stage_index + 1)
        if len(self.conversation_history) < need:
            logging.error(
                f"Cannot build single-stage transcript for task {self.scenario_id} "
                f"(Iter {self.iteration_index}): need {need} messages, have "
                f"{len(self.conversation_history)}."
            )
            return None
        user_msg = self.conversation_history[2 * stage_index]
        asst_msg = self.conversation_history[2 * stage_index + 1]
        if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
            logging.error(
                f"Unexpected message roles at stage {stage_index} for {self.scenario_id}: "
                f"{user_msg.get('role')}, {asst_msg.get('role')}"
            )
            return None
        user_content = user_msg.get("content", "")
        assistant_processed = self._format_assistant_content_for_rubric(
            stage_index, truncate_for_rubric
        )
        parts = [
            f"User:\n{user_content}\n",
            f"Assistant:\n{assistant_processed}\n",
        ]
        return "---\n".join(parts)

    def run_turn_rubric(
        self,
        api_clients: Dict[str, Any],
        rubric_prompt_template: str,
        rubric_output_format_str: str,
        turn_index: int,
        judge_models: List[str],
    ):
        transcript = self.build_single_stage_rubric_transcript(
            stage_index=turn_index,
            truncate_for_rubric=True,
        )
        if transcript is None:
            logging.warning(
                f"Skipping turn rubric for task {self.scenario_id} turn {turn_index}: "
                "could not build single-stage transcript."
            )
            return

        prompt = rubric_prompt_template.format(
            transcript=transcript,
            output_format=rubric_output_format_str,
        )

        judge_api = api_clients.get("judge")
        if not judge_api:
            logging.warning(
                f"No judge API client for turn rubric (task {self.scenario_id}, turn {turn_index})"
            )
            return

        def _one(mid: str) -> str:
            return judge_api.generate(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

        raws: List[str] = []
        if len(judge_models) == 1:
            raws.append(_one(judge_models[0]))
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(judge_models), 4),
                thread_name_prefix="TurnRubric",
            ) as tp:
                futures = [tp.submit(_one, mid) for mid in judge_models]
                for fut in futures:
                    raws.append(fut.result())

        parsed_list: List[Dict[str, float]] = []
        for raw in raws:
            s = self._parse_rubric_scores(raw.strip())
            if s is None:
                logging.warning(
                    f"Turn rubric parse failed for {self.scenario_id} (Iter {self.iteration_index}) turn {turn_index}"
                )
                return
            parsed_list.append(s)

        try:
            scores = aggregate_rubric_scores(parsed_list)
        except ValueError as e:
            logging.warning(
                f"Turn rubric aggregate failed for {self.scenario_id} turn {turn_index}: {e}"
            )
            return

        combined_raw = "\n---\n".join(r.strip() for r in raws)

        while len(self.turn_rubric_scores) <= turn_index:
            self.turn_rubric_scores.append(None)
            self.turn_raw_rubric_judge_text.append(None)

        self.turn_rubric_scores[turn_index] = scores
        self.turn_raw_rubric_judge_text[turn_index] = combined_raw

        while len(self.turn_rubric_scores_by_judge) <= turn_index:
            self.turn_rubric_scores_by_judge.append(None)
            self.turn_raw_rubric_judge_text_by_judge.append(None)
        self.turn_rubric_scores_by_judge[turn_index] = [dict(d) for d in parsed_list]
        self.turn_raw_rubric_judge_text_by_judge[turn_index] = list(raws)

    @staticmethod
    def _transcript_from_messages(
        messages: List[Dict[str, str]], truncate_for_rubric: bool
    ) -> str:
        del truncate_for_rubric  # reserved for parity with turn transcript path
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}:\n{content}\n")
        return "---\n".join(parts)

    def run_baseline_rubric(
        self,
        api_clients: Dict[str, Any],
        rubric_prompt_template: str,
        rubric_output_format_str: str,
        judge_models: List[str],
    ):
        if not self.baseline_conversation_history:
            return
        if self.baseline_conversation_history[-1]["role"] != "assistant":
            return
        transcript = self._transcript_from_messages(
            self.baseline_conversation_history, truncate_for_rubric=True
        )
        prompt = rubric_prompt_template.format(
            transcript=transcript,
            output_format=rubric_output_format_str,
        )
        judge_api = api_clients.get("judge")
        if not judge_api:
            logging.warning(
                f"No judge API client for baseline rubric (task {self.scenario_id})"
            )
            return

        def _one(mid: str) -> str:
            return judge_api.generate(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

        raws: List[str] = []
        if len(judge_models) == 1:
            raws.append(_one(judge_models[0]))
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(judge_models), 4),
                thread_name_prefix="BaselineRubric",
            ) as tp:
                futures = [tp.submit(_one, mid) for mid in judge_models]
                for fut in futures:
                    raws.append(fut.result())

        parsed_list: List[Dict[str, float]] = []
        for raw in raws:
            s = self._parse_rubric_scores(raw.strip())
            if s is None:
                logging.warning(
                    f"Baseline rubric parse failed for {self.scenario_id} (Iter {self.iteration_index})"
                )
                return
            parsed_list.append(s)

        try:
            scores = aggregate_rubric_scores(parsed_list)
        except ValueError as e:
            logging.warning(
                f"Baseline rubric aggregate failed for {self.scenario_id}: {e}"
            )
            return

        self.baseline_rubric_scores = scores
        self.baseline_raw_rubric_judge_text = "\n---\n".join(r.strip() for r in raws)
        self.baseline_rubric_scores_by_judge = [dict(d) for d in parsed_list]
        self.baseline_raw_rubric_judge_text_by_judge = list(raws)

    def run_baseline_commitment_rubric(
        self,
        api_clients: Dict[str, Any],
        rubric_prompt_template: str,
        rubric_output_format_str: str,
        judge_models: List[str],
    ):
        """Store commitment (0–5) in baseline_commitment_* (dual-scoring runs)."""
        if not self.baseline_conversation_history:
            return
        if self.baseline_conversation_history[-1]["role"] != "assistant":
            return
        transcript = self._transcript_from_messages(
            self.baseline_conversation_history, truncate_for_rubric=True
        )
        prompt = rubric_prompt_template.format(
            transcript=transcript,
            output_format=rubric_output_format_str,
        )
        judge_api = api_clients.get("judge")
        if not judge_api:
            logging.warning(
                f"No judge API client for baseline commitment (task {self.scenario_id})"
            )
            return

        def _one(mid: str) -> str:
            return judge_api.generate(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

        raws: List[str] = []
        if len(judge_models) == 1:
            raws.append(_one(judge_models[0]))
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(judge_models), 4),
                thread_name_prefix="BaselineCommitment",
            ) as tp:
                futures = [tp.submit(_one, mid) for mid in judge_models]
                for fut in futures:
                    raws.append(fut.result())

        parsed_list: List[Dict[str, float]] = []
        for raw in raws:
            s = self._parse_rubric_scores(raw.strip())
            if s is None:
                logging.warning(
                    f"Baseline commitment parse failed for {self.scenario_id} (Iter {self.iteration_index})"
                )
                return
            parsed_list.append(s)

        try:
            scores = aggregate_rubric_scores(parsed_list)
        except ValueError as e:
            logging.warning(
                f"Baseline commitment aggregate failed for {self.scenario_id}: {e}"
            )
            return

        self.baseline_commitment_scores = scores
        self.baseline_raw_commitment_judge_text = "\n---\n".join(
            r.strip() for r in raws
        )
        self.baseline_commitment_scores_by_judge = [dict(d) for d in parsed_list]
        self.baseline_raw_commitment_judge_text_by_judge = list(raws)

    def run_turn_commitment_rubric(
        self,
        api_clients: Dict[str, Any],
        rubric_prompt_template: str,
        rubric_output_format_str: str,
        turn_index: int,
        judge_models: List[str],
    ):
        """Store commitment scores in turn_commitment_* (dual-scoring runs)."""
        transcript = self.build_single_stage_rubric_transcript(
            stage_index=turn_index,
            truncate_for_rubric=True,
        )
        if transcript is None:
            logging.warning(
                f"Skipping turn commitment rubric for task {self.scenario_id} turn {turn_index}: "
                "could not build single-stage transcript."
            )
            return

        prompt = rubric_prompt_template.format(
            transcript=transcript,
            output_format=rubric_output_format_str,
        )

        judge_api = api_clients.get("judge")
        if not judge_api:
            logging.warning(
                f"No judge API client for turn commitment (task {self.scenario_id}, turn {turn_index})"
            )
            return

        def _one(mid: str) -> str:
            return judge_api.generate(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

        raws: List[str] = []
        if len(judge_models) == 1:
            raws.append(_one(judge_models[0]))
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(judge_models), 4),
                thread_name_prefix="TurnCommitment",
            ) as tp:
                futures = [tp.submit(_one, mid) for mid in judge_models]
                for fut in futures:
                    raws.append(fut.result())

        parsed_list: List[Dict[str, float]] = []
        for raw in raws:
            s = self._parse_rubric_scores(raw.strip())
            if s is None:
                logging.warning(
                    f"Turn commitment parse failed for {self.scenario_id} "
                    f"(Iter {self.iteration_index}) turn {turn_index}"
                )
                return
            parsed_list.append(s)

        try:
            scores = aggregate_rubric_scores(parsed_list)
        except ValueError as e:
            logging.warning(
                f"Turn commitment aggregate failed for {self.scenario_id} turn {turn_index}: {e}"
            )
            return

        combined_raw = "\n---\n".join(r.strip() for r in raws)

        while len(self.turn_commitment_scores) <= turn_index:
            self.turn_commitment_scores.append(None)
            self.turn_raw_commitment_judge_text.append(None)

        self.turn_commitment_scores[turn_index] = scores
        self.turn_raw_commitment_judge_text[turn_index] = combined_raw

        while len(self.turn_commitment_scores_by_judge) <= turn_index:
            self.turn_commitment_scores_by_judge.append(None)
            self.turn_raw_commitment_judge_text_by_judge.append(None)
        self.turn_commitment_scores_by_judge[turn_index] = [dict(d) for d in parsed_list]
        self.turn_raw_commitment_judge_text_by_judge[turn_index] = list(raws)

    def _run_baseline_turn(
        self,
        test_api: Any,
        api_model_id: str,
        api_clients: Dict[str, Any],
        judge_models: Optional[List[str]],
        save_queue: Optional[queue.Queue],
        run_key: Optional[str],
        trait_turn_template: Optional[str] = None,
        trait_turn_format: Optional[str] = None,
        commitment_turn_template: Optional[str] = None,
        commitment_turn_format: Optional[str] = None,
        trait_judging: bool = True,
        commitment_judging: bool = True,
        commitment_only_storage: bool = False,
    ) -> None:
        if not self.baseline_prompt:
            return
        user_content = self.BASELINE_USER_PREFIX + self.baseline_prompt
        if not self.baseline_conversation_history:
            self.baseline_conversation_history = [{"role": "user", "content": user_content}]
        elif self.baseline_conversation_history[0].get("content") != user_content:
            logging.warning(
                "Baseline prompt text changed vs saved history for %s (iter %s); "
                "keeping existing baseline messages.",
                self.scenario_id,
                self.iteration_index,
            )

        msgs = list(self.baseline_conversation_history)
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
            pass
        elif len(msgs) == 1 and msgs[-1]["role"] == "user":
            assistant_response = test_api.generate(
                model=api_model_id,
                messages=msgs,
                temperature=0.7,
                max_tokens=8000,
                min_p=0.1,
            )
            self.baseline_conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )
            self.baseline_parsed_response = {"raw": assistant_response}
        else:
            logging.warning(
                "Unexpected baseline_conversation_history for %s: len=%s last_role=%s",
                self.scenario_id,
                len(msgs),
                msgs[-1]["role"] if msgs else None,
            )
            return

        if judge_models and len(self.baseline_conversation_history) >= 2:
            trait_tmpl = trait_turn_template or TURN_RUBRIC_PROMPT_TEMPLATE
            trait_fmt = trait_turn_format or TURN_RUBRIC_OUTPUT_FORMAT
            if trait_judging and self.baseline_rubric_scores is None:
                self.run_baseline_rubric(
                    api_clients=api_clients,
                    rubric_prompt_template=trait_tmpl,
                    rubric_output_format_str=trait_fmt,
                    judge_models=judge_models,
                )
            if commitment_judging and commitment_turn_template and commitment_turn_format:
                if commitment_only_storage:
                    if self.baseline_rubric_scores is None:
                        self.run_baseline_rubric(
                            api_clients=api_clients,
                            rubric_prompt_template=commitment_turn_template,
                            rubric_output_format_str=commitment_turn_format,
                            judge_models=judge_models,
                        )
                elif self.baseline_commitment_scores is None:
                    self.run_baseline_commitment_rubric(
                        api_clients=api_clients,
                        rubric_prompt_template=commitment_turn_template,
                        rubric_output_format_str=commitment_turn_format,
                        judge_models=judge_models,
                    )
        self._save_progress(save_queue, run_key)

    def run_scenario(
        self,
        api_clients: Dict[str, Any],
        save_queue: Optional[queue.Queue],
        run_key: Optional[str],
        api_model_id: str,
        judge_models: Optional[List[str]] = None,
        trait_turn_rubric_template: Optional[str] = None,
        trait_turn_rubric_output_format_str: Optional[str] = None,
        commitment_turn_rubric_template: Optional[str] = None,
        commitment_turn_rubric_output_format_str: Optional[str] = None,
        trait_judging: bool = True,
        commitment_judging: bool = True,
        commitment_only_storage: bool = False,
    ):
        """
        Runs the multi-turn scenario simulation sequentially using the test model.
        Updates conversation_history and status. Puts progress updates on the save_queue.
        Handles standard, drafting, NO_RP, and analysis types.
        Uses the provided api_model_id for API calls.
        """
        is_analysis = self.scenario_id in ANALYSIS_SCENARIO_IDS
        # Determine valid skip statuses based on type
        valid_skip_statuses = ["scenario_completed", "running_rubric_scoring", "rubric_scored"]
        if not is_analysis:
            valid_skip_statuses.extend(["running_debrief", "completed"])

        if self.status in valid_skip_statuses:
            logging.debug(f"Scenario task {self.scenario_id} (Iter {self.iteration_index}) status is '{self.status}'. Skipping scenario run.")
            return
        if self.status == "error":
             logging.info(f"Retrying scenario run for task {self.scenario_id} (Iter {self.iteration_index}) which previously errored.")

        logging.info(f"Starting scenario run for task {self.scenario_id} (Iter {self.iteration_index}, Analysis: {is_analysis}) with model {self.model_name} (API ID: {api_model_id})")
        self.status = "running_scenario"
        self.scenario_run_error = None; self.error = None
        if not self.start_time: self.start_time = time.time()

        test_api = api_clients["test"]
        if self.conversation_history and self.scenario_run_error: # Check specific error flag
            logging.info(f"Clearing previous history for errored task {self.scenario_id} (Iter {self.iteration_index}) before retry.")
            self.conversation_history = []; self.parsed_responses = []
            self.baseline_conversation_history = []
            self.baseline_rubric_scores = None
            self.baseline_raw_rubric_judge_text = None
            self.baseline_rubric_scores_by_judge = None
            self.baseline_raw_rubric_judge_text_by_judge = None
            self.baseline_commitment_scores = None
            self.baseline_raw_commitment_judge_text = None
            self.baseline_commitment_scores_by_judge = None
            self.baseline_raw_commitment_judge_text_by_judge = None
            self.baseline_parsed_response = None

        current_messages = list(self.conversation_history)
        turn_index = -1 # Initialize turn index

        try:
            self._run_baseline_turn(
                test_api=test_api,
                api_model_id=api_model_id,
                api_clients=api_clients,
                judge_models=judge_models,
                save_queue=save_queue,
                run_key=run_key,
                trait_turn_template=trait_turn_rubric_template,
                trait_turn_format=trait_turn_rubric_output_format_str,
                commitment_turn_template=commitment_turn_rubric_template,
                commitment_turn_format=commitment_turn_rubric_output_format_str,
                trait_judging=trait_judging,
                commitment_judging=commitment_judging,
                commitment_only_storage=commitment_only_storage,
            )
            for turn_index, user_prompt in enumerate(self.prompts):
                expected_len_after_this_turn = (turn_index + 1) * 2
                if len(current_messages) >= expected_len_after_this_turn:
                    logging.debug(f"Skipping already completed turn {turn_index + 1} for scenario {self.scenario_id} (Iter {self.iteration_index})")
                    continue

                logging.debug(f"Running turn {turn_index + 1}/{len(self.prompts)} for scenario {self.scenario_id} (Iter {self.iteration_index})")
                no_rp_scenario = self.scenario_id in NO_RP_SCENARIO_IDS

                # Apply master prompt template if provided and applicable
                if self.master_prompt_template and not no_rp_scenario:
                    # Master template is already selected based on type in benchmark.py
                    formatted_prompt = self.master_prompt_template.format(scenario_prompt=user_prompt)
                else:
                    # Fallback for NO_RP or if template is missing (shouldn't happen for analysis/drafting now)
                    formatted_prompt = user_prompt # Maybe add word count hint here too?

                # Add user prompt to history
                if not current_messages or current_messages[-1]["role"] == "assistant":
                    current_messages.append({"role": "user", "content": formatted_prompt})
                elif current_messages[-1]["role"] == "user" and current_messages[-1]["content"] != formatted_prompt:
                     logging.warning(f"Overwriting last user message at turn {turn_index+1} for {self.scenario_id} (Iter {self.iteration_index}).")
                     current_messages[-1] = {"role": "user", "content": formatted_prompt}
                # Handle case where resuming exactly after user prompt was added but before assistant response
                elif current_messages[-1]["role"] == "user" and current_messages[-1]["content"] == formatted_prompt:
                     pass # Correct state, proceed to generate assistant response
                else: # Should not happen
                     logging.error(f"Unexpected history state at turn {turn_index+1} for {self.scenario_id}. Last msg: {current_messages[-1]['role']}")
                     # Attempt to recover by adding user prompt if missing
                     if current_messages[-1]["role"] != "user":
                         current_messages.append({"role": "user", "content": formatted_prompt})


                assistant_response = test_api.generate(
                    model=api_model_id, # Use the API model ID here
                    messages=current_messages,
                    temperature=0.7, # Consider different temps per type?
                    max_tokens=12000, # Consider different lengths per type?
                    min_p=0.1
                    )

                # Store response: Parse for standard/drafting, store raw for NO_RP/Analysis
                parsed_entry = {}
                if not no_rp_scenario and not is_analysis:
                    parsed_entry = self._parse_response(assistant_response)
                else:
                    # Store raw response in a consistent structure
                    parsed_entry = {"raw": assistant_response}

                # Ensure parsed_responses list is updated correctly
                if len(self.parsed_responses) == turn_index:
                    self.parsed_responses.append(parsed_entry)
                elif len(self.parsed_responses) > turn_index:
                    self.parsed_responses[turn_index] = parsed_entry # Overwrite if resuming/rerunning
                else:
                    # This indicates a logic error, list should grow sequentially
                    logging.error(f"Inconsistent parsed_responses length for {self.scenario_id} (Iter {self.iteration_index}) at turn {turn_index+1}. List len: {len(self.parsed_responses)}")
                    # Attempt recovery: append anyway?
                    self.parsed_responses.append(parsed_entry)


                # Add assistant response to history
                if len(current_messages) == (turn_index * 2) + 1: # Should be after user prompt
                    current_messages.append({"role": "assistant", "content": assistant_response})
                elif current_messages[-1]["role"] == "user": # Also valid if resuming exactly here
                     current_messages.append({"role": "assistant", "content": assistant_response})
                else: # Should not happen
                     logging.error(f"Unexpected history state before adding assistant response at turn {turn_index+1} for {self.scenario_id}. Last msg: {current_messages[-1]['role']}")
                     # Attempt to recover
                     current_messages.append({"role": "assistant", "content": assistant_response})

                self.conversation_history = list(current_messages)

                if judge_models:
                    trait_tmpl = trait_turn_rubric_template or TURN_RUBRIC_PROMPT_TEMPLATE
                    trait_fmt = (
                        trait_turn_rubric_output_format_str or TURN_RUBRIC_OUTPUT_FORMAT
                    )
                    if trait_judging:
                        need_trait = turn_index >= len(self.turn_rubric_scores) or (
                            self.turn_rubric_scores[turn_index] is None
                        )
                        if need_trait:
                            self.run_turn_rubric(
                                api_clients=api_clients,
                                rubric_prompt_template=trait_tmpl,
                                rubric_output_format_str=trait_fmt,
                                turn_index=turn_index,
                                judge_models=judge_models,
                            )
                    if commitment_judging and commitment_turn_rubric_template and commitment_turn_rubric_output_format_str:
                        if commitment_only_storage:
                            need_c = turn_index >= len(self.turn_rubric_scores) or (
                                self.turn_rubric_scores[turn_index] is None
                            )
                            if need_c:
                                self.run_turn_rubric(
                                    api_clients=api_clients,
                                    rubric_prompt_template=commitment_turn_rubric_template,
                                    rubric_output_format_str=commitment_turn_rubric_output_format_str,
                                    turn_index=turn_index,
                                    judge_models=judge_models,
                                )
                        else:
                            need_c = turn_index >= len(
                                self.turn_commitment_scores
                            ) or (self.turn_commitment_scores[turn_index] is None)
                            if need_c:
                                self.run_turn_commitment_rubric(
                                    api_clients=api_clients,
                                    rubric_prompt_template=commitment_turn_rubric_template,
                                    rubric_output_format_str=commitment_turn_rubric_output_format_str,
                                    turn_index=turn_index,
                                    judge_models=judge_models,
                                )

                self._save_progress(save_queue, run_key)

            # Scenario completed successfully
            self.status = "scenario_completed" # Correct status for all types after scenario run
            logging.info(f"Scenario run finished successfully for task {self.scenario_id} (Iter {self.iteration_index})")

        except Exception as e:
            current_turn_for_error = turn_index + 1 if turn_index >= 0 else 1 # Adjust error reporting turn
            error_msg = f"Scenario Run Error at turn {current_turn_for_error}: {str(e)}"
            logging.error(f"Error during scenario run for task {self.scenario_id} (Iter {self.iteration_index}) at turn {current_turn_for_error}: {e}", exc_info=True)
            self.status = "error"; self.error = error_msg; self.scenario_run_error = str(e)
        finally:
            self._save_progress(save_queue, run_key)


    def run_debrief(self, api_clients: Dict[str, Any], save_queue: Optional[queue.Queue], run_key: Optional[str], api_model_id: str):
        """
        Runs the debrief prompt using the test model after the scenario is completed.
        SKIPS this step for Analysis tasks.
        Updates debrief_response and status. Puts progress updates on the save_queue.
        Uses the provided api_model_id for API calls.
        """
        # --- Skip entirely for Analysis tasks ---
        if self.scenario_id in ANALYSIS_SCENARIO_IDS:
            logging.debug(f"Skipping debrief step for analysis task {self.scenario_id} (Iter {self.iteration_index}). Status remains '{self.status}'.")
            # No status change needed, stays 'scenario_completed' ready for rubric
            return

        # --- Logic for non-analysis tasks ---
        if self.status not in ["scenario_completed", "error"]:
             logging.debug(f"Debrief cannot run for task {self.scenario_id} (Iter {self.iteration_index}). Status is '{self.status}'. Skipping.")
             return
        # If errored during scenario, skip debrief unless it was specifically a debrief error we are retrying
        if self.status == "error" and not self.debrief_run_error:
             logging.warning(f"Task {self.scenario_id} (Iter {self.iteration_index}) errored before debrief. Skipping debrief.")
             # Don't save progress here, error state is already saved
             return
        if self.status == "error" and self.debrief_run_error:
             logging.info(f"Retrying debrief run for task {self.scenario_id} (Iter {self.iteration_index}) which previously errored during debrief.")

        # If debrief already done (e.g., resuming), move to 'completed'
        if self.debrief_response is not None and self.status == "scenario_completed":
             logging.info(f"Debrief response already exists for task {self.scenario_id} (Iter {self.iteration_index}). Marking as completed (ready for rubric).")
             self.status = "completed"
             self._save_progress(save_queue, run_key)
             return
        # If already past debrief stage
        if self.status in ["completed", "running_rubric_scoring", "rubric_scored"]:
             logging.debug(f"Debrief already completed or passed for task {self.scenario_id} (Iter {self.iteration_index}). Status: '{self.status}'. Skipping.")
             return

        logging.info(f"Starting debrief run for task {self.scenario_id} (Iter {self.iteration_index})")
        self.status = "running_debrief"
        self.debrief_run_error = None; self.error = None # Clear previous errors for retry

        test_api = api_clients["test"]
        if not self.conversation_history:
             logging.error(f"Cannot run debrief for task {self.scenario_id} (Iter {self.iteration_index}): Conversation history is empty.")
             self.status = "error"; self.error = "Debrief Run Error: Conversation history is empty."; self.debrief_run_error = "Conversation history is empty."
             self._save_progress(save_queue, run_key)
             return
        if not self.debrief_prompt: # Should not happen for non-analysis tasks if loaded correctly
             logging.error(f"Cannot run debrief for task {self.scenario_id} (Iter {self.iteration_index}): Debrief prompt is missing.")
             self.status = "error"; self.error = "Debrief Run Error: Debrief prompt missing."; self.debrief_run_error = "Debrief prompt missing."
             self._save_progress(save_queue, run_key)
             return


        debrief_messages = list(self.conversation_history)
        debrief_messages.append({"role": "user", "content": self.debrief_prompt})

        try:
            response = test_api.generate(
                model=api_model_id, # Use the API model ID here
                messages=debrief_messages,
                temperature=0.5,
                max_tokens=12000,
                min_p=None
            )
            self.debrief_response = response.strip()
            self.status = "completed" # Final status before optional rubric step
            logging.info(f"Debrief run finished successfully for task {self.scenario_id} (Iter {self.iteration_index})")
        except Exception as e:
            error_msg = f"Debrief Run Error: {str(e)}"
            logging.error(f"Error during debrief run for task {self.scenario_id} (Iter {self.iteration_index}): {e}", exc_info=True)
            self.status = "error"; self.error = error_msg; self.debrief_run_error = str(e)
        finally:
            self._save_progress(save_queue, run_key)


    # --- Helper Methods for Truncation (Static) ---
    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        """Truncates text to a character limit."""
        if not text or len(text) <= limit:
            return text
        return text[:limit] + "... [truncated]"

    @staticmethod
    def _truncate_parsed_sections(parsed: Dict[str, str], limits: Dict[str, int]) -> Dict[str, str]:
        """Truncate each section of parsed response according to character limits."""
        truncated = {}
        # Ensure all expected keys from limits are present, even if empty or not in parsed
        for section_name in limits:
            content = parsed.get(section_name) # Use .get for safety
            if content:
                char_limit = limits[section_name]
                truncated[section_name] = ScenarioTask._truncate_text(content, char_limit)
            else:
                truncated[section_name] = "" # Keep empty content as empty string
        return truncated


    @staticmethod
    def _format_parsed_response(sections: Dict[str, str]) -> str:
        """Render parsed sections back into a readable block."""
        formatted = []
        # Check for role-play keys first
        if "thinking_feeling" in sections:
            formatted.append("# I'm thinking & feeling")
            formatted.append(sections.get("thinking_feeling", ""))
            formatted.append("\n# They're thinking & feeling")
            formatted.append(sections.get("their_thinking_feeling", ""))
            formatted.append("\n# My response")
            formatted.append(sections.get("response", ""))
        # Check for message-drafting keys
        elif "perspective_taking" in sections:
            formatted.append("# Perspective-taking")
            formatted.append(sections.get("perspective_taking", ""))
            formatted.append("\n# Draft brainstorming")
            formatted.append(sections.get("draft_brainstorming", ""))
            formatted.append("\n# Draft")
            formatted.append(sections.get("draft", ""))
        else: # Fallback if keys don't match expected patterns
             logging.warning("Could not format parsed response: Unknown section keys.")
             # Attempt to join non-empty values from known section types
             known_keys = list(SECTION_CHAR_LIMITS.keys()) + list(SECTION_CHAR_LIMITS_MESSAGE_DRAFT.keys())
             content_parts = [v for k, v in sections.items() if k in known_keys and v]
             return "\n\n".join(content_parts)

        # Add the standard separator if sections were found
        if formatted:
            formatted.append("\n--")
        return "\n".join(formatted)


    # --- NEW Rubric Helper Methods ---

    def prepare_rubric_prompt_text(
        self,
        rubric_prompt_template: str, # Specific template for this task type
        rubric_output_format_str: str, # Specific format for this task type
        truncate_for_rubric: bool # Added flag
    ) -> Optional[str]:
        """
        Formats the rubric scoring prompt. Scenario roleplay is judged per stage during
        run_scenario (run_turn_rubric); this final prompt uses debrief-only text for
        standard tasks, or the last analysis stage only for analysis tasks.
        """
        is_analysis = self.scenario_id in ANALYSIS_SCENARIO_IDS

        if not self.conversation_history:
            logging.error(f"Cannot prepare rubric prompt for task {self.scenario_id} (Iter {self.iteration_index}): Conversation history is empty.")
            self.status = "error"; self.error = "Rubric Prep Error: History missing."; self.rubric_run_error = "History missing."
            return None
        # Debrief is required ONLY for non-analysis tasks
        if not is_analysis and self.debrief_response is None:
            logging.error(f"Cannot prepare rubric prompt for task {self.scenario_id} (Iter {self.iteration_index}): Debrief response is missing for non-analysis task.")
            self.status = "error"; self.error = "Rubric Prep Error: Debrief missing."; self.rubric_run_error = "Debrief missing."
            return None

        # Roleplay is omitted here for standard tasks (each stage scored in isolation in run_turn_rubric).
        # Analysis: judge sees only the last stage (typically the full analysis output).
        if is_analysis:
            n_stages = len(self.conversation_history) // 2
            if n_stages < 1:
                logging.error(
                    f"Cannot prepare rubric prompt for task {self.scenario_id} (Iter {self.iteration_index}): "
                    "no complete user/assistant turns in history."
                )
                self.status = "error"
                self.error = "Rubric Prep Error: No turns in history."
                self.rubric_run_error = "No turns in history."
                return None
            last_stage_index = n_stages - 1
            full_transcript = self.build_single_stage_rubric_transcript(
                last_stage_index, truncate_for_rubric
            )
            if full_transcript is None:
                self.status = "error"
                self.error = "Rubric Prep Error: Could not build analysis transcript."
                self.rubric_run_error = "Analysis transcript build failed."
                return None
        else:
            full_transcript = (
                "[The scripted multi-turn roleplay is omitted: each stage was scored separately. "
                "Do not infer values from omitted roleplay. Score only the debrief below on the rubric criteria.]"
            )

        # Handle debrief text (only relevant for non-analysis)
        debrief_text = ""
        if not is_analysis:
            debrief_text = self.debrief_response or "" # Use empty string if None somehow
            if truncate_for_rubric and len(debrief_text) > DEBRIEF_CHAR_LIMIT:
                debrief_text = debrief_text[:DEBRIEF_CHAR_LIMIT] + '...[truncated]'

        # Format the final prompt using the specific template
        try:
            format_args = {
                "transcript": full_transcript,
                "output_format": rubric_output_format_str
            }
            # Only add debrief if it's expected by the template (i.e., not analysis)
            if not is_analysis:
                format_args["debrief"] = debrief_text

            final_rubric_prompt = rubric_prompt_template.format(**format_args)

            #print(final_rubric_prompt)

            return final_rubric_prompt
        except KeyError as e:
             # This error means the template has a placeholder not provided in format_args
             logging.error(f"Missing key '{e}' in rubric prompt template formatting for task {self.scenario_id} (Iter {self.iteration_index}, Analysis: {is_analysis}). Provided keys: {list(format_args.keys())}")
             self.status = "error"; self.error = f"Rubric Prep Error: Invalid template key ({e})."; self.rubric_run_error = f"Invalid template key ({e})."
             return None
        except Exception as e: # Catch other formatting errors
             logging.error(f"Error formatting rubric prompt for task {self.scenario_id} (Iter {self.iteration_index}): {e}", exc_info=True)
             self.status = "error"; self.error = "Rubric Prep Error: Formatting failed."; self.rubric_run_error = f"Formatting failed: {e}"
             return None

    def process_rubric_response(self, raw_judge_response: str):
        """
        Processes the raw response from the judge model for rubric scoring.
        Parses scores, updates status, scores, errors, and end time.
        """
        self.raw_rubric_judge_text = raw_judge_response
        parsed_scores = self._parse_rubric_scores(raw_judge_response)

        if parsed_scores is not None:
            self.rubric_scores = parsed_scores
            self.rubric_scores_by_judge = None
            self.raw_rubric_judge_text_by_judge = None
            self.status = "rubric_scored" # Final success state for rubric
            self.end_time = time.time() # Mark final completion time
            self.error = None # Clear general error if scoring succeeded
            self.rubric_run_error = None # Clear specific rubric error
            logging.debug(f"Rubric scoring processed successfully for task {self.scenario_id} (Iter {self.iteration_index})")
        else:
            # Parsing failed
            error_msg = "Rubric Scoring Error: Failed to parse scores from judge response."
            logging.error(f"{error_msg} Task: {self.scenario_id} (Iter {self.iteration_index})")
            self.status = "error"
            self.error = error_msg
            self.rubric_run_error = "Failed to parse scores"
            self.rubric_scores = None # Ensure scores are None
            self.rubric_scores_by_judge = None
            self.raw_rubric_judge_text_by_judge = None


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the task state to a dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "prompts": self.prompts,
            "debrief_prompt": self.debrief_prompt, # Can be None
            "iteration_index": self.iteration_index,
            "test_model": self.model_name, # Serialize logical name into 'test_model' key
            "master_prompt_template": self.master_prompt_template, # Can be None if not used
            "baseline_prompt": self.baseline_prompt,
            "baseline_conversation_history": self.baseline_conversation_history,
            "baseline_rubric_scores": self.baseline_rubric_scores,
            "baseline_raw_rubric_judge_text": self.baseline_raw_rubric_judge_text,
            "baseline_parsed_response": self.baseline_parsed_response,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            # Specific errors
            "scenario_run_error": self.scenario_run_error,
            "debrief_run_error": self.debrief_run_error,
            "rubric_run_error": self.rubric_run_error,
            # Data payloads
            "conversation_history": self.conversation_history,
            "parsed_responses": self.parsed_responses,
            "debrief_response": self.debrief_response, # Can be None
            "rubric_scores": self.rubric_scores, # Can be None
            "raw_rubric_judge_text": self.raw_rubric_judge_text, # Can be None,
            "rubric_scores_by_judge": self.rubric_scores_by_judge,
            "raw_rubric_judge_text_by_judge": self.raw_rubric_judge_text_by_judge,
            "turn_rubric_scores": self.turn_rubric_scores,
            "turn_raw_rubric_judge_text": self.turn_raw_rubric_judge_text,
            "turn_rubric_scores_by_judge": self.turn_rubric_scores_by_judge,
            "turn_raw_rubric_judge_text_by_judge": self.turn_raw_rubric_judge_text_by_judge,
            "baseline_rubric_scores_by_judge": self.baseline_rubric_scores_by_judge,
            "baseline_raw_rubric_judge_text_by_judge": self.baseline_raw_rubric_judge_text_by_judge,
            "baseline_commitment_scores": self.baseline_commitment_scores,
            "baseline_raw_commitment_judge_text": self.baseline_raw_commitment_judge_text,
            "baseline_commitment_scores_by_judge": self.baseline_commitment_scores_by_judge,
            "baseline_raw_commitment_judge_text_by_judge": self.baseline_raw_commitment_judge_text_by_judge,
            "turn_commitment_scores": self.turn_commitment_scores,
            "turn_raw_commitment_judge_text": self.turn_raw_commitment_judge_text,
            "turn_commitment_scores_by_judge": self.turn_commitment_scores_by_judge,
            "turn_raw_commitment_judge_text_by_judge": self.turn_raw_commitment_judge_text_by_judge,
            "debrief_commitment_scores": self.debrief_commitment_scores,
            "raw_debrief_commitment_judge_text": self.raw_debrief_commitment_judge_text,
            "debrief_commitment_scores_by_judge": self.debrief_commitment_scores_by_judge,
            "raw_debrief_commitment_judge_text_by_judge": self.raw_debrief_commitment_judge_text_by_judge,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Deserializes a task state from a dictionary."""
        # Debrief prompt is now optional
        # 'test_model' key now holds the logical name
        required_keys = ["scenario_id", "prompts", "iteration_index", "test_model"]
        if not all(key in data for key in required_keys):
             missing = [k for k in required_keys if k not in data]
             raise ValueError(f"Missing required keys in task data dictionary: {missing}. Available keys: {list(data.keys())}")

        obj = cls(
            scenario_id=data["scenario_id"],
            prompts=data["prompts"],
            debrief_prompt=data.get("debrief_prompt"), # Use .get, allows None
            iteration_index=data.get("iteration_index", 0), # Keep default for older data
            test_model=data["test_model"], # Read logical name from 'test_model' key
            master_prompt_template=data.get("master_prompt_template"), # Use .get
            baseline_prompt=data.get("baseline_prompt"),
        )
        obj.status = data.get("status", "initialized")
        obj.start_time = data.get("start_time")
        obj.end_time = data.get("end_time")
        obj.error = data.get("error")
        # Specific errors
        obj.scenario_run_error = data.get("scenario_run_error")
        obj.debrief_run_error = data.get("debrief_run_error")
        obj.rubric_run_error = data.get("rubric_run_error")
        # Data payloads
        obj.conversation_history = data.get("conversation_history", [])
        obj.parsed_responses = data.get("parsed_responses", [])
        obj.debrief_response = data.get("debrief_response")
        obj.rubric_scores = data.get("rubric_scores")
        obj.raw_rubric_judge_text = data.get("raw_rubric_judge_text")
        obj.rubric_scores_by_judge = data.get("rubric_scores_by_judge")
        obj.raw_rubric_judge_text_by_judge = data.get("raw_rubric_judge_text_by_judge")
        obj.turn_rubric_scores = data.get("turn_rubric_scores", [])
        obj.turn_raw_rubric_judge_text = data.get("turn_raw_rubric_judge_text", [])
        obj.turn_rubric_scores_by_judge = data.get("turn_rubric_scores_by_judge") or []
        obj.turn_raw_rubric_judge_text_by_judge = data.get(
            "turn_raw_rubric_judge_text_by_judge"
        ) or []
        obj.baseline_conversation_history = data.get("baseline_conversation_history") or []
        obj.baseline_rubric_scores = data.get("baseline_rubric_scores")
        obj.baseline_raw_rubric_judge_text = data.get("baseline_raw_rubric_judge_text")
        obj.baseline_rubric_scores_by_judge = data.get("baseline_rubric_scores_by_judge")
        obj.baseline_raw_rubric_judge_text_by_judge = data.get(
            "baseline_raw_rubric_judge_text_by_judge"
        )
        obj.baseline_parsed_response = data.get("baseline_parsed_response")
        obj.baseline_commitment_scores = data.get("baseline_commitment_scores")
        obj.baseline_raw_commitment_judge_text = data.get(
            "baseline_raw_commitment_judge_text"
        )
        obj.baseline_commitment_scores_by_judge = data.get(
            "baseline_commitment_scores_by_judge"
        )
        obj.baseline_raw_commitment_judge_text_by_judge = data.get(
            "baseline_raw_commitment_judge_text_by_judge"
        )
        obj.turn_commitment_scores = data.get("turn_commitment_scores", [])
        obj.turn_raw_commitment_judge_text = data.get(
            "turn_raw_commitment_judge_text", []
        )
        obj.turn_commitment_scores_by_judge = data.get(
            "turn_commitment_scores_by_judge"
        ) or []
        obj.turn_raw_commitment_judge_text_by_judge = data.get(
            "turn_raw_commitment_judge_text_by_judge"
        ) or []
        obj.debrief_commitment_scores = data.get("debrief_commitment_scores")
        obj.raw_debrief_commitment_judge_text = data.get(
            "raw_debrief_commitment_judge_text"
        )
        obj.debrief_commitment_scores_by_judge = data.get(
            "debrief_commitment_scores_by_judge"
        )
        obj.raw_debrief_commitment_judge_text_by_judge = data.get(
            "raw_debrief_commitment_judge_text_by_judge"
        )
        return obj