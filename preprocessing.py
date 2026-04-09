"""
Preprocessing module for alignment drift datasets.

Loads scenario JSON files, concatenates conversation turns, tokenizes using
model-specific tokenizers, and saves preprocessed tensors with metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import torch
from transformers import AutoTokenizer
import re

# Configuration
DATA_DIR = Path(__file__).parent / "dataset"  # Data files in dataset folder
PREPROCESSED_DIR = Path(__file__).parent / "preprocessed"

# Model configurations: (HuggingFace ID, separator token, max_length)
MODEL_CONFIG = {
    "bart": ("facebook/bart-large", "</s>", 1024),
    "t5": ("google-t5/t5-base", "<sep>", 512),
    "pegasus": ("google/pegasus-large", "<n>", 1024),
}

SCENARIO_IDS = ["A", "B", "C", "D", "E"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_scenario_json(scenario_id: str) -> Dict:
    """Load a scenario JSON file from the data directory.
    
    Args:
        scenario_id (str): Scenario identifier (A-E).
    
    Returns:
        Dict: Parsed JSON content.
    
    Raises:
        FileNotFoundError: If scenario file doesn't exist.
        json.JSONDecodeError: If JSON is malformed.
    
    Example:
        >>> data = load_scenario_json("A")
        >>> len(data["conversations"])
        10
    """
    scenario_map = {
        "A": "scenario_A_instruction_override.json",
        "B": "scenario_B_emotional_manipulation.json",
        "C": "scenario_C_over_agreeableness.json",
        "D": "scenario_D_gradual_context_shift.json",
        "E": "scenario_E_memory_stress.json",
    }
    
    file_path = DATA_DIR / scenario_map[scenario_id]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded scenario {scenario_id} from {file_path}")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON from {file_path}: {e.msg}",
            e.doc,
            e.pos
        )


def build_conversation_string(
    turns: List[Dict],
    separator: str,
    up_to_turn: int
) -> str:
    """Build conversation string up to a specific turn.
    
    Concatenates turns in the format:
    [ROLE]: content
    [ROLE]: content
    ...
    
    Args:
        turns (List[Dict]): List of turn dictionaries with 'role' and 'content'.
        separator (str): Token/string to separate turns.
        up_to_turn (int): Don't include turns at or after this index (1-indexed).
    
    Returns:
        str: Concatenated conversation string.
    
    Example:
        >>> turns = [
        ...     {"role": "system", "content": "Be helpful"},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> result = build_conversation_string(turns, "</s>", 3)
        >>> "[SYSTEM]: Be helpful</s>[USER]: Hello" in result
        True
    """
    conversation_parts = []
    
    # Include all turns with index < up_to_turn (1-indexed, turn field starts at 1)
    for turn in turns:
        turn_idx = turn.get("turn", 0)
        if turn_idx >= up_to_turn:
            break
        
        role = turn.get("role", "").upper()
        content = turn.get("content", "")
        
        turn_str = f"[{role}]: {content}"
        conversation_parts.append(turn_str)
    
    # Join with separator
    conversation_string = separator.join(conversation_parts)
    return conversation_string


def find_probe_turns(turns: List[Dict]) -> List[int]:
    """Find indices of probe turns (marked with label='EVAL').
    
    Args:
        turns (List[Dict]): List of turn dictionaries.
    
    Returns:
        List[int]: List of probe turn indices (1-indexed turn numbers).
    
    Example:
        >>> turns = [
        ...     {"turn": 1, "role": "system", "label": None},
        ...     {"turn": 7, "role": "assistant", "label": "EVAL"}
        ... ]
        >>> find_probe_turns(turns)
        [7]
    """
    probe_turns = []
    for turn in turns:
        if turn.get("label") == "EVAL":
            probe_turns.append(turn.get("turn", -1))
    return probe_turns


def tokenize_and_truncate(
    text: str,
    tokenizer,
    max_length: int
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Tokenize text and truncate to max_length.
    
    Args:
        text (str): Input text to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Maximum token length.
    
    Returns:
        Tuple containing:
            - input_ids (torch.Tensor): Token IDs.
            - attention_mask (torch.Tensor): Attention mask.
            - original_length (int): Original token count.
            - truncated (bool): Whether truncation occurred.
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        >>> ids, mask, orig_len, trunc = tokenize_and_truncate("Hello", tokenizer, 10)
        >>> ids.shape[0] <= 10
        True
    """
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=False,  # First pass: no truncation to measure original
        padding=False,
        return_tensors=None  # Get list-based output
    )
    
    original_length = len(encoding["input_ids"])
    
    # Now truncate if needed
    if original_length > max_length:
        truncated = True
        input_ids = encoding["input_ids"][:max_length]
        attention_mask = encoding["attention_mask"][:max_length]
    else:
        truncated = False
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
    
    # Convert to tensors
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    
    return input_ids_tensor, attention_mask_tensor, original_length, truncated


def process_conversation(
    conv_id: str,
    turns: List[Dict],
    scenario_id: str,
    model_id: str,
    tokenizer,
    separator: str,
    max_length: int
) -> Tuple[List[Dict], List[Dict]]:
    """Process a single conversation into preprocessed tensors.
    
    Identifies probe turns and builds context strings for each.
    
    Args:
        conv_id (str): Conversation identifier.
        turns (List[Dict]): List of turn dictionaries.
        scenario_id (str): Scenario identifier.
        model_id (str): Model identifier (bart/t5/pegasus).
        tokenizer: HuggingFace tokenizer.
        separator (str): Turn separator token.
        max_length (int): Max token length.
    
    Returns:
        Tuple of:
            - saved_files (List[Dict]): List of saved file info dicts.
            - logs (List[Dict]): List of preprocessing log entries.
    
    Example:
        >>> turns = [{"turn": 1, "role": "system", "content": "Be helpful", "label": None}]
        >>> results, logs = process_conversation("A-001", turns, "A", "bart", tok, "</s>", 1024)
    """
    probe_turns = find_probe_turns(turns)
    saved_files = []
    logs = []
    
    for probe_turn_idx in probe_turns:
        # Build context string up to (but not including) probe turn
        context_string = build_conversation_string(turns, separator, probe_turn_idx)
        
        # Tokenize
        input_ids, attention_mask, orig_len, truncated = tokenize_and_truncate(
            context_string,
            tokenizer,
            max_length
        )
        
        # Create output directory if needed
        output_dir = PREPROCESSED_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tensor
        output_file = output_dir / f"{scenario_id}_{conv_id}_probe{probe_turn_idx}.pt"
        torch.save(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "conv_id": conv_id,
                "scenario_id": scenario_id,
                "probe_turn": probe_turn_idx,
            },
            output_file
        )
        
        saved_files.append({
            "file": str(output_file),
            "conv_id": conv_id,
            "probe_turn": probe_turn_idx
        })
        
        # Log
        logs.append({
            "model": model_id,
            "scenario_id": scenario_id,
            "conv_id": conv_id,
            "probe_turn": probe_turn_idx,
            "original_token_count": orig_len,
            "truncated_token_count": min(orig_len, max_length),
            "truncation_occurred": truncated,
        })
    
    return saved_files, logs


def preprocess_scenario(
    scenario_id: str,
    model_id: str,
    tokenizer,
    separator: str,
    max_length: int
) -> Dict:
    """Preprocess all conversations in a scenario for a specific model.
    
    Args:
        scenario_id (str): Scenario identifier.
        model_id (str): Model identifier.
        tokenizer: HuggingFace tokenizer.
        separator (str): Turn separator.
        max_length (int): Max token length.
    
    Returns:
        Dict with keys 'files_saved' and 'logs'.
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        >>> result = preprocess_scenario("A", "bart", tokenizer, "</s>", 1024)
    """
    try:
        data = load_scenario_json(scenario_id)
    except Exception as e:
        logger.warning(f"Failed to load scenario {scenario_id}: {e}")
        return {"files_saved": [], "logs": []}
    
    all_saved_files = []
    all_logs = []
    
    conversations = data.get("conversations", [])
    logger.info(f"Processing {len(conversations)} conversations for scenario {scenario_id}, model {model_id}")
    
    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("conv_id", f"{scenario_id}-{conv_idx:03d}")
        turns = conv.get("conversations", [])
        
        saved_files, logs = process_conversation(
            conv_id=conv_id,
            turns=turns,
            scenario_id=scenario_id,
            model_id=model_id,
            tokenizer=tokenizer,
            separator=separator,
            max_length=max_length
        )
        
        all_saved_files.extend(saved_files)
        all_logs.extend(logs)
        
        if (conv_idx + 1) % 5 == 0:
            logger.info(f"  Processed {conv_idx + 1}/{len(conversations)} conversations")
    
    logger.info(f"Completed scenario {scenario_id}, model {model_id}: {len(all_logs)} probe turns saved")
    
    return {
        "files_saved": all_saved_files,
        "logs": all_logs
    }


def save_preprocessing_log(all_logs: List[Dict], model_id: str) -> None:
    """Save preprocessing log to JSON.
    
    Args:
        all_logs (List[Dict]): List of log entries.
        model_id (str): Model identifier.
    
    Example:
        >>> logs = [{"model": "bart", "scenario_id": "A", ...}]
        >>> save_preprocessing_log(logs, "bart")
    """
    log_file = PREPROCESSED_DIR / "preprocessing_log.json"
    
    existing_logs = []
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                existing_logs = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing log file {log_file}, starting fresh")
    
    # Append new logs for this model
    combined_logs = existing_logs + all_logs
    
    with open(log_file, "w") as f:
        json.dump(combined_logs, f, indent=2)
    
    logger.info(f"Saved preprocessing log to {log_file} ({len(combined_logs)} total entries)")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess alignment drift datasets"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="all",
        choices=["bart", "t5", "pegasus", "all"],
        help="Model to preprocess (default: all)"
    )
    parser.add_argument(
        "--scenario_id",
        type=str,
        default="all",
        choices=["A", "B", "C", "D", "E", "all"],
        help="Scenario to preprocess (default: all)"
    )
    
    args = parser.parse_args()
    
    # Determine models and scenarios to process
    models_to_process = ["bart", "t5", "pegasus"] if args.model_id == "all" else [args.model_id]
    scenarios_to_process = ["A", "B", "C", "D", "E"] if args.scenario_id == "all" else [args.scenario_id]
    
    logger.info(f"Starting preprocessing: models={models_to_process}, scenarios={scenarios_to_process}")
    
    all_preprocessing_logs = []
    
    for model_id in models_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_id}")
        logger.info(f"{'='*60}")
        
        model_name, separator, max_length = MODEL_CONFIG[model_id]
        logger.info(f"Loading tokenizer: {model_name}")
        
        try:
            # For PEGASUS, use slow tokenizer due to sentencepiece compatibility
            if model_id == "pegasus":
                logger.info("PEGASUS requires sentencepiece. Attempting to load...")
                try:
                    import sentencepiece
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                except ImportError:
                    logger.warning("⚠️  PEGASUS skipped: 'sentencepiece' not available")
                    logger.warning("   To fix: brew install protobuf && pip install sentencepiece")
                    continue
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            if model_id == "pegasus":
                logger.warning(f"⚠️  PEGASUS skipped: {type(e).__name__}")
                logger.warning("   To fix: brew install protobuf && pip install sentencepiece")
            else:
                logger.error(f"Failed to load tokenizer for {model_id}: {e}")
            continue
        
        for scenario_id in scenarios_to_process:
            result = preprocess_scenario(
                scenario_id=scenario_id,
                model_id=model_id,
                tokenizer=tokenizer,
                separator=separator,
                max_length=max_length
            )
            
            all_preprocessing_logs.extend(result["logs"])
    
    # Save unified log
    save_preprocessing_log(all_preprocessing_logs, args.model_id)
    
    logger.info(f"\n{'='*60}")
    logger.info("Preprocessing complete!")
    logger.info(f"Total probe turns processed: {len(all_preprocessing_logs)}")
    logger.info(f"Preprocessed files saved to: {PREPROCESSED_DIR}")
    logger.info(f"Log file: {PREPROCESSED_DIR / 'preprocessing_log.json'}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
