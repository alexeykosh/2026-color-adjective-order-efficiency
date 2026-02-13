"""
Extract sentences without color adjectives and add properly inflected color adjectives.

This script:
1. Loads sentence pairs with color adjectives (from preprocess_colors.py output)
2. Finds sentences in Wikipedia that contain the same nouns but WITHOUT adjectives
3. Inserts adjective placeholders in the correct position
4. Uses an LLM (Qwen) to inflect adjectives for proper agreement

Usage:
    python add_color_adjectives.py [--n-samples N] [--model MODEL_ID] [--no-gpu]

Output:
    sentence_pairs_color_adj_added.pkl - Pairs of (sentence_without_adj, sentence_with_adj)
"""

import random
import pickle
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from preprocess_colors import ColorPreprocessor


# =============================================================================
# Configuration
# =============================================================================

LANGUAGES = {
    "de": "German",
    "es": "Spanish", 
    "fr": "French",
    "ru": "Russian"
}

WIKI_FILES = {
    'de': 'wikipedia_german_300k.txt',
    'es': 'wikipedia_spanish_300k.txt',
    'fr': 'wikipedia_french_300k.txt',
    'ru': 'wikipedia_russian_300k.txt',
}

# Adjective position by language
HEAD_SIDE = {"fr": "left", "es": "left", "de": "right", "ru": "right"}


# =============================================================================
# LLM Inflection Functions
# =============================================================================

SYSTEM_PROMPT = (
    "You are a deterministic morphology engine. "
    "Output exactly ONE word: the correctly inflected adjective form. "
    "No punctuation. No extra text."
)

CORE_PROMPT = """Task:
You receive ONE sentence with exactly ONE adjective placeholder:
<<ADJ<lemma=LEMMA>>> or <<ADJ<lemma=LEMMA;...features...>>>

HARD CONSTRAINT (dataset guarantee):
- For [fr, es]: the head noun is the closest NOUN immediately to the LEFT of the placeholder. (NOUN <<ADJ>>)
- For [de, ru]: the head noun is the closest NOUN immediately to the RIGHT of the placeholder. (<<ADJ>> NOUN)
Ignore all other nouns.

Agreement:
Inflect the adjective to match the head noun in all relevant categories for the language:
gender, number, case, definiteness/declension class, animacy (if applicable).
Respect any explicit features inside the placeholder if they do not contradict the head noun.

Output:
Return EXACTLY ONE word: the inflected adjective. No punctuation. No extra text.
"""

LANG_RULES = {
    "fr": """French notes:
- Use nearby determiners (un/une/le/la/du/de la/des/les) as strongest cues for gender/number.
""",
    "es": """Spanish notes:
- Inflect for gender/number of the head noun.
- Apocope if the adjective is immediately before a masc sg noun:
  bueno→buen, malo→mal, primero→primer, tercero→tercer, alguno→algún, ninguno→ningún
""",
    "de": """German notes:
- Attributive adjectives inflect for case/gender/number and strong/weak/mixed depending on article.
- Predicative adjectives (after sein/werden/bleiben) are typically uninflected.
""",
    "ru": """Russian notes:
- Inflect for gender/number/case; consider animacy in accusative.
- Prefer long-form adjectives unless context clearly requires short form.
""",
}

EXAMPLES = """Examples:
[fr] Sentence: un député <<ADJ<lemma=vert>>> .
Answer: vert
[es] Sentence: una casa <<ADJ<lemma=blanco>>> .
Answer: blanca
[de] Sentence: <<ADJ<lemma=klein>>> Haus .
Answer: kleines
[ru] Sentence: <<ADJ<lemma=новый>>> книга .
Answer: новая
"""

_LEGACY_RE = re.compile(r"<<ADJ<(?P<inner>[^>]+)>>>")
_WORD_RE = re.compile(r"[^\W\d_]+(?:[-''][^\W\d_]+)*|\d+", flags=re.UNICODE)


def normalize_placeholder(sentence: str) -> str:
    """Normalize adjective placeholders to standard format."""
    if "<<ADJ<lemma=" in sentence:
        return sentence

    def repl(m: re.Match) -> str:
        inner = m.group("inner").strip()
        if inner.startswith("lemma="):
            return f"<<ADJ<{inner}>>>"
        if ";" in inner:
            head, rest = inner.split(";", 1)
            return f"<<ADJ<lemma={head.strip()};{rest.strip()}>>>"
        return f"<<ADJ<lemma={inner}>>>"

    return _LEGACY_RE.sub(repl, sentence)


def nearest_word_left(text: str) -> str:
    toks = _WORD_RE.findall(text)
    return toks[-1] if toks else ""


def nearest_word_right(text: str) -> str:
    toks = _WORD_RE.findall(text)
    return toks[0] if toks else ""


def extract_head_hint(sentence: str, lang: str) -> str:
    """Extract the head noun hint based on language-specific position."""
    side = HEAD_SIDE.get(lang, "left")
    parts = re.split(r"<<ADJ<lemma=[^>]+>>>", sentence, maxsplit=1)
    if len(parts) != 2:
        return ""
    left_text, right_text = parts[0], parts[1]
    return nearest_word_left(left_text) if side == "left" else nearest_word_right(right_text)


def build_prompt(lang: str, sentence: str) -> str:
    """Build the LLM prompt for adjective inflection."""
    sentence = normalize_placeholder(sentence)
    rules = LANG_RULES.get(lang, "")
    head_hint = extract_head_hint(sentence, lang)
    hint = f"Head noun (adjacent, per dataset): {head_hint}\n" if head_hint else ""
    return f"""{CORE_PROMPT}{rules}{EXAMPLES}
{hint}Now do:
[{lang}] Sentence: {sentence}
Answer:"""


def inflect_adjective(model, tokenizer, lang: str, sentence: str, max_new_tokens: int = 6) -> str:
    """Use LLM to inflect adjective for proper agreement."""
    user_prompt = build_prompt(lang, sentence)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    gen = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    word = gen.strip().split()[0].strip(" \t\n\r\"'""''.,;:!?)(")
    return word


# =============================================================================
# Sentence Extraction Functions
# =============================================================================

def load_color_adj_data(data_dir: Path) -> dict:
    """Load sentence pairs with color adjectives from preprocess_colors.py output."""
    all_data = {}
    for lang_code, lang_name in LANGUAGES.items():
        pickle_file = data_dir / "processed" / f"{lang_code}_sentence_pairs_color_adj.pkl"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                all_data[lang_code] = pickle.load(f)
            print(f"Loaded {lang_name}: {len(all_data[lang_code])} pairs")
        else:
            print(f"⚠ {lang_name} file not found: {pickle_file}")
    return all_data


def build_collocations(all_data: dict) -> dict:
    """Build most frequent noun-adjective collocations from data."""
    most_freq_collocations = {}
    for lang, rows in all_data.items():
        counts = defaultdict(Counter)
        for row in rows:
            if len(row) < 4:
                continue
            adj = str(row[2]).strip()
            noun = str(row[3]).strip()
            if not adj or not noun:
                continue
            counts[noun][adj] += 1
        most_freq_collocations[lang] = {
            noun: c.most_common(1)[0][0] 
            for noun, c in counts.items() if c
        }
    return most_freq_collocations


def extract_nonadj_sentences(data_dir: Path, collocations: dict, n_samples: int = 400) -> dict:
    """Extract sentences with target nouns but without adjectives."""
    results = {}
    
    for lang_code, wiki_file in WIKI_FILES.items():
        print(f"\nExtracting {LANGUAGES[lang_code]} sentences...")
        input_path = data_dir / wiki_file
        
        if not input_path.exists():
            print(f"  ⚠ Wikipedia file not found: {input_path}")
            continue
        
        if lang_code not in collocations:
            print(f"  ⚠ No collocations for {lang_code}")
            continue
            
        preproc = ColorPreprocessor(lang_code, str(data_dir), verbose=False)
        target_nouns = set(collocations[lang_code].keys())
        dep_label = preproc.adj_dep_label
        
        # Load and shuffle Wikipedia lines
        with open(input_path, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        random.shuffle(lines)
        
        # Extract candidate sentences
        candidates = []
        for line in lines:
            for sent in preproc.sentence_splitter.split(line):
                sent = re.sub(r'\s+', ' ', sent.replace('\xa0', ' ')).strip()
                if (preproc.valid_sentence_regex.match(sent) and 
                    not preproc.junk_pattern.search(sent) and
                    preproc.sentence_end_pattern.search(sent)):
                    candidates.append(sent)
                    if len(candidates) >= n_samples * 20:
                        break
            if len(candidates) >= n_samples * 20:
                break
        
        # Find sentences with target nouns without adjectives
        pairs = []
        used_sentences = set()
        
        for doc, sent in zip(preproc.nlp.pipe(candidates, batch_size=100), candidates):
            if sent in used_sentences:
                continue
                
            for token in doc:
                if (token.pos_ == 'NOUN' and 
                    token.lemma_ in target_nouns and
                    not any(c.pos_ == 'ADJ' and c.dep_ == dep_label for c in token.children)):
                    
                    adj = collocations[lang_code][token.lemma_]
                    tokens = [t.text for t in doc]
                    adj_marker = f"<<ADJ<{adj}>>>"
                    
                    # Insert adjective marker in correct position
                    if lang_code in ['de', 'ru']:
                        tokens.insert(token.i, adj_marker)
                    else:
                        tokens.insert(token.i + 1, adj_marker)
                    
                    masked_sent = ' '.join(tokens)
                    
                    if lang_code in ['de', 'ru']:
                        pair = f"{adj} {token.text}"
                    else:
                        pair = f"{token.text} {adj}"
                    
                    pairs.append([pair, masked_sent])
                    used_sentences.add(sent)
                    break
                    
            if len(pairs) >= n_samples:
                break
        
        results[lang_code] = pairs
        print(f"  {len(pairs)} pairs extracted")
    
    return results


def inflect_all_adjectives(nonadj_sentences: dict, model, tokenizer) -> dict:
    """Inflect all adjective placeholders using LLM."""
    results = {}
    
    for lang, pairs in nonadj_sentences.items():
        print(f"\nInflecting {LANGUAGES[lang]} adjectives...")
        results[lang] = []
        
        for item in tqdm(pairs, desc=f"  {lang}"):
            sent = item[1]
            sent_norm = normalize_placeholder(sent)
            
            # Get inflected adjective
            pred = inflect_adjective(model, tokenizer, lang, sent_norm)
            
            # Create sentence without adjective
            sent_without = re.sub(r'<<ADJ<[^>]+>>>', '', sent_norm)
            sent_without = re.sub(r'\s+', ' ', sent_without).strip()
            
            # Create sentence with inflected adjective
            sent_with = re.sub(r'<<ADJ<[^>]+>>>', pred, sent_norm, count=1)
            
            results[lang].append((sent_without, sent_with))
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract non-adjective sentences and add inflected color adjectives"
    )
    parser.add_argument(
        "--n-samples", type=int, default=400,
        help="Number of sentence pairs per language (default: 400)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID for inflection (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU-only inference"
    )
    parser.add_argument(
        "--output", type=str, default="sentence_pairs_color_adj_added.pkl",
        help="Output pickle file (default: sentence_pairs_color_adj_added.pkl)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Step 1: Load color adjective data
    print("=" * 60)
    print("Step 1: Loading color adjective data")
    print("=" * 60)
    all_data = load_color_adj_data(data_dir)
    
    if not all_data:
        print("No data found. Run preprocess_colors.py first.")
        return
    
    # Step 2: Build collocations
    print("\n" + "=" * 60)
    print("Step 2: Building noun-adjective collocations")
    print("=" * 60)
    collocations = build_collocations(all_data)
    for lang, cols in collocations.items():
        print(f"  {LANGUAGES[lang]}: {len(cols)} noun types")
    
    # Step 3: Extract non-adjective sentences
    print("\n" + "=" * 60)
    print("Step 3: Extracting sentences without adjectives")
    print("=" * 60)
    nonadj_sentences = extract_nonadj_sentences(data_dir, collocations, args.n_samples)
    
    # Step 4: Load LLM for inflection
    print("\n" + "=" * 60)
    print("Step 4: Loading LLM for adjective inflection")
    print("=" * 60)
    print(f"Model: {args.model}")
    
    device_map = "cpu" if args.no_gpu else "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=device_map,
    )
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Step 5: Inflect adjectives
    print("\n" + "=" * 60)
    print("Step 5: Inflecting adjectives")
    print("=" * 60)
    results = inflect_all_adjectives(nonadj_sentences, model, tokenizer)
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {args.output}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for lang, pairs in results.items():
        print(f"  {LANGUAGES[lang]}: {len(pairs)} sentence pairs")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
