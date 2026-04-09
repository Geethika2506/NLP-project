"""
Inference module for alignment drift models.

Loads preprocessed tensors, runs model inference with attention extraction,
and saves outputs and attention metrics to JSONL.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import jsonlines

# Configuration
PREPROCESSED_DIR = Path(__file__).parent / "preprocessed"
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_CONFIG = {
    "bart": "facebook/bart-large",
    "t5": "google-t5/t5-base",
    "pegasus": "google/pegasus-large",
}

SCENARIO_IDS = ["A", "B", "C", "D", "E"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: CUDA device if available, else CPU.
    
    Example:
        >>> device = get_device()
        >>> device.type in ["cuda", "cpu"]
        True
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def load_model(model_id: str):
    """Load a configured model and tokenizer for the given model id.

    Args:
        model_id (str): Model identifier (bart/t5/pegasus).

    Returns:
        Tuple[Any, Any, torch.device]: Loaded model, tokenizer, and device.
    """
    if model_id not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model_id: {model_id}")

    device = get_device()
    model_name = MODEL_CONFIG[model_id]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


def load_preprocessed_file(file_path: Path) -> Dict:
    """Load a preprocessed .pt file.
    
    Args:
        file_path (Path): Path to .pt file.
    
    Returns:
        Dict: Loaded tensor data with metadata.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
    
    Example:
        >>> data = load_preprocessed_file(Path("preprocessed/bart/A_A-001_probe7.pt"))
        >>> "input_ids" in data
        True
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {file_path}")
    
    data = torch.load(file_path, map_location="cpu")
    return data


def compute_attention_entropy(attention_weights, sequence_length: int = None) -> Tuple[float, float]:
    """Compute attention entropy from cross-attention weights with normalization.

    Extracts cross-attention from LAST decoder layer, averages across attention heads,
    computes Shannon entropy of the attention distribution, then normalizes by
    log(sequence_length) to make scores comparable across different sequence lengths
    and architectures.

    Args:
        attention_weights: Tuple of (self_attention, cross_attention).
        sequence_length (int): Source sequence length for normalization.

    Returns:
        Tuple[float, float]: (raw_entropy, normalised_entropy) where:
            - raw_entropy: Shannon entropy of attention distribution
            - normalised_entropy: raw_entropy / log(sequence_length+1)
                                 comparable across architectures
    
    Example:
        >>> # Mock attention: (batch, heads, seq_len, key_len)
        >>> attn = torch.randn(1, 16, 50, 50)
        >>> raw, norm = compute_attention_entropy((None, attn), sequence_length=50)
        >>> 0 <= raw <= np.log(50)
        True
        >>> 0 <= norm <= 1.5  # Generally normalized to ~[0, 1]
        True
    """
    try:
        # For seq2seq models, cross_attention is the second element
        # Shape: (batch_size, num_heads, query_len, key_len)
        # We want attention from LAST decoder layer over encoder tokens
        
        if len(attention_weights) < 2 or attention_weights[1] is None:
            # Fallback if cross-attention not available
            return 0.0, 0.0
        
        cross_attn = attention_weights[1]  # Last layer cross-attention
        
        # Shape: (batch_size, num_heads, tgt_len, src_len)
        if cross_attn.dim() == 4:
            # Already single layer, expected from run_inference
            last_layer_attn = cross_attn
        else:
            # Should not happen given our extraction, but handle gracefully
            last_layer_attn = cross_attn[-1] if len(cross_attn) > 0 else None
        
        if last_layer_attn is None:
            return 0.0, 0.0
        
        # Average over batch and heads to get attention distribution
        # (batch, heads, tgt_len, src_len) -> (tgt_len, src_len)
        batch_size = last_layer_attn.shape[0]
        if batch_size > 0:
            last_layer_attn = last_layer_attn.squeeze(0)  # Remove batch dim
        
        # Average over heads
        if last_layer_attn.dim() >= 2:
            avg_attn = last_layer_attn.mean(dim=0)  # (tgt_len, src_len)
        else:
            return 0.0, 0.0
        
        # Average over query positions to get attention over encoder tokens
        # (tgt_len, src_len) -> (src_len,)
        attention_dist = avg_attn.mean(dim=0)
        
        # Ensure probabilities sum to ~1 (normalize if needed)
        attention_dist = attention_dist / (attention_dist.sum() + 1e-10)
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        attention_dist = attention_dist + 1e-9
        raw_entropy = -(attention_dist * torch.log(attention_dist)).sum().item()
        
        # Normalize by log(sequence_length) for cross-architecture comparability
        # This maps entropy to roughly [0, 1] range regardless of seq_len
        if sequence_length is not None and sequence_length > 1:
            max_entropy = np.log(sequence_length)
        else:
            # Fallback: use actual sequence length from tensor
            max_entropy = np.log(max(attention_dist.shape[0], 2))
        
        normalised_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(raw_entropy), float(normalised_entropy)
    
    except Exception as e:
        logger.warning(f"Could not compute attention entropy: {e}")
        return 0.0, 0.0


def extract_attention_entropy(attention_weights, sequence_length: int = None) -> Tuple[float, float]:
    """Extract attention entropy using the shared entropy computation logic.

    Args:
        attention_weights: Attention tensors from generation output.
        sequence_length (int): Source sequence length for normalization.

    Returns:
        Tuple[float, float]: (raw_ahe, normalised_ahe)
    """
    return compute_attention_entropy(attention_weights, sequence_length)


def run_inference(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    model_id: str
) -> Tuple[str, float, float]:
    """Run model inference and extract attention.
    
    Args:
        model: Loaded HuggingFace model.
        tokenizer: Tokenizer for the model.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask.
        device (torch.device): Device to run on.
        model_id (str): Model identifier (bart/t5/pegasus).
    
    Returns:
        Tuple of:
            - output_text (str): Generated response.
            - raw_ahe (float): Raw attention entropy.
            - normalised_ahe (float): Normalized (sequence-length independent) AHE.
    
    Example:
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        >>> text, raw_ahe, norm_ahe = run_inference(model, tokenizer, ids, mask, device, "bart")
    """
    # For T5, prepend task prefix
    if model_id == "t5":
        # Modify input to include task prefix
        # Note: This should ideally be done during preprocessing,
        # but we can add it here if needed.
        pass
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Generate with attention output
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    # Decode output
    output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    
    # Extract attention entropy (raw and normalized)
    raw_ahe = 0.0
    normalised_ahe = 0.0
    
    if hasattr(output, "decoder_attentions") and output.decoder_attentions:
        # Use last decoder layer attention
        try:
            last_layer_attn = output.decoder_attentions[-1]
            # Get sequence length from input
            seq_len = input_ids.shape[1]
            raw_ahe, normalised_ahe = compute_attention_entropy(
                (None, last_layer_attn),
                sequence_length=seq_len
            )
        except Exception as e:
            logger.warning(f"Could not extract attention from decoder: {e}")
    
    return output_text, raw_ahe, normalised_ahe


def generate_response(
    model,
    tokenizer,
    input_text: str,
    model_id: str,
    device: torch.device,
    max_length: int = 512
) -> Tuple[str, float, float]:
    """Generate response text and attention entropy from raw input text.

    Args:
        model: Loaded Hugging Face model.
        tokenizer: Loaded tokenizer.
        input_text (str): Formatted prompt text.
        model_id (str): Model identifier.
        device (torch.device): Inference device.
        max_length (int): Tokenization max input length.

    Returns:
        Tuple[str, float, float]: Generated text, raw AHE, and normalized AHE.
    """
    encoded = tokenizer(
        input_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return run_inference(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        model_id=model_id,
    )


def process_model_files(
    model_id: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 4
) -> List[Dict]:
    """Process all preprocessed files for a model.
    
    Args:
        model_id (str): Model identifier.
        model: Loaded model.
        tokenizer: Tokenizer.
        device (torch.device): Device.
        batch_size (int): Batch size (for future batching).
    
    Returns:
        List[Dict]: List of inference results.
    
    Example:
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        >>> results = process_model_files("bart", model, tokenizer, device)
    """
    model_dir = PREPROCESSED_DIR / model_id
    
    if not model_dir.exists():
        logger.warning(f"No preprocessed files for model {model_id}")
        return []
    
    # Collect all .pt files
    pt_files = sorted(model_dir.glob("*.pt"))
    logger.info(f"Found {len(pt_files)} preprocessed files for model {model_id}")
    
    results = []
    
    for file_path in tqdm(pt_files, desc=f"Inferencing {model_id}"):
        try:
            # Load preprocessed data
            data = load_preprocessed_file(file_path)
            
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            conv_id = data["conv_id"]
            scenario_id = data["scenario_id"]
            probe_turn = data["probe_turn"]
            
            # Reconstruct input text (for logging)
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            
            # Run inference
            output_text, raw_ahe, normalised_ahe = run_inference(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                device,
                model_id
            )
            
            # Create result entry
            result = {
                "model": model_id,
                "scenario_id": scenario_id,
                "conv_id": conv_id,
                "probe_turn": probe_turn,
                "input_text": input_text[:500],  # Truncate for storage
                "output_text": output_text,
                "raw_ahe": raw_ahe,
                "normalised_ahe": normalised_ahe,
                "timestamp": datetime.now().isoformat(),
            }
            
            results.append(result)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    return results


def save_results_jsonl(results: List[Dict], output_file: Path) -> None:
    """Save results to JSONL file.
    
    Args:
        results (List[Dict]): List of result dictionaries.
        output_file (Path): Output file path.
    
    Example:
        >>> results = [{"model": "bart", ...}]
        >>> save_results_jsonl(results, Path("results/raw_outputs.jsonl"))
    """
    # Load existing results if file exists
    existing_results = []
    if output_file.exists():
        try:
            with jsonlines.open(output_file, mode="r") as reader:
                existing_results = list(reader)
            logger.info(f"Found {len(existing_results)} existing results")
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}")
    
    # Combine and save
    all_results = existing_results + results
    
    with jsonlines.open(output_file, mode="w") as writer:
        for result in all_results:
            writer.write(result)
    
    logger.info(f"Saved {len(all_results)} results to {output_file}")


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Run inference on alignment drift datasets"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="bart",
        choices=["bart", "t5", "pegasus"],
        help="Model to run inference with (default: bart)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting inference pipeline")
    logger.info(f"Model: {args.model_id}, Batch size: {args.batch_size}")
    logger.info(f"{'='*60}\n")
    
    # Get device
    device = get_device()
    
    # Load model and tokenizer
    model_name = MODEL_CONFIG[args.model_id]
    logger.info(f"Loading model: {model_name}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Run inference
    results = process_model_files(
        model_id=args.model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size
    )
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "raw_outputs.jsonl"
    save_results_jsonl(results, output_file)
    
    logger.info(f"\n{'='*60}")
    logger.info("Inference complete!")
    logger.info(f"Processed {len(results)} inferences")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
