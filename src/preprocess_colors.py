"""
Multi-language color adjective preprocessing script.

This script extracts sentences with color adjectives from Wikipedia dumps
and creates sentence pairs with/without color adjectives for analysis.

Supports: German, Spanish, French, Russian
"""

import re
import pickle
import argparse
import unicodedata
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import spacy

# Language-specific configurations
LANG_CONFIG = {
    "de": {
        "spacy_model": "de_core_news_lg",
        "adj_dep_label": {"amod", "nk"},  # German uses 'nk' (noun kernel modifier) instead of 'amod'
        "colors": ["gelb", "blau", "weiß", "grau", "braun", 
                   "violett", "orange", "schwarz", "rot", "rosa", "grün"],
        "valid_chars_regex": r"^[\w\s.,;:!?äöüßÄÖÜ]+$"
    },
    "es": {
        "spacy_model": "es_core_news_lg",
        "adj_dep_label": "amod",
        "colors": ["amarillo", "azul", "blanco", "gris", "marrón", 
                   "morado", "naranja", "negro", "rojo", "rosa", "verde"],
        "valid_chars_regex": r"^[\w\s.,;:!?¿¡]+$"
    },
    "fr": {
        "spacy_model": "fr_core_news_lg",
        "adj_dep_label": "amod",
        "colors": ["jaune", "bleu", "blanc", "gris", "marron", 
                   "violet", "orange", "noir", "rouge", "rose", "vert"],
        "valid_chars_regex": r"^[\w\s.,;:!?àâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ]+$"
    },
    "ru": {
        "spacy_model": "ru_core_news_lg",
        "adj_dep_label": "amod",
        "colors": ["жёлтый", "синий", "белый", "серый", "коричневый", 
                   "фиолетовый", "оранжевый", "чёрный", "красный", "розовый", "зелёный"],
        "valid_chars_regex": r"^[\w\s.,;:!?А-Яа-яЁё]+$"
    }
}


def build_is_junk_sentence(lang_code: str):
    """
    Returns a fast is_junk_sentence(sent)->bool function tuned for Wikipedia-ish text.
    Safe to use on WikiExtractor output, or any 'mostly plain text but still noisy' dumps.
    """

    # Common Wikipedia/meta keywords (keep fairly conservative)
    # Add/remove terms based on what you see in your corpus.
    META_PATTERNS = [
        r"\bISBN\b",
        r"\bISSN\b",
        r"\bDOI\b",
        r"\bPMID\b",
        r"\barXiv\b",
        r"\bSBN\b",
        r"\bOCLC\b",
        r"\bBibcode\b",
        r"\bpp\.\b",
        r"\bvol\.\b",
        r"\bNo\.\b",
        r"\bed\.\b",
        r"\bet al\.\b",
        r"\bRetrieved\b|\bAccessed\b|\bArchived\b|\barchive(?:d)?\b",
        r"\bCategory\b|\bKategorie\b|\bCategor(?:ía|ia)\b|\bКатегория\b",
        r"\bFile\b|\bDatei\b|\bFichier\b|\bФайл\b|\bImage\b|\bBild\b",
        r"\bExternal links\b|\bWeblinks\b|\bLiens externes\b|\bВнешние ссылки\b",
        r"\bReferences\b|\bEinzelnachweise\b|\bRéférences\b|\bПримечания\b|\bИсточники\b",
        r"\bSee also\b|\bSiehe auch\b|\bVoir aussi\b|\bСм\. также\b",
        r"\bCoordinates\b|\bKoordinaten\b|\bCoordonnées\b|\bКоординаты\b",
    ]

    # URLs / HTML entities / tags
    URL_OR_HTML = re.compile(
        r"(https?://|www\.)|(&[a-zA-Z]+;)|(<[^>]+>)",
        flags=re.IGNORECASE
    )

    # Residual wiki markup (often removed upstream, but sometimes leaks)
    WIKI_MARKUP = re.compile(r"(\{\{|\}\}|\[\[|\]\]|==|__\w+__|\|\})")

    # Citation-like brackets or ref leftovers
    REF_LIKE = re.compile(r"(\[\d+\])|(\(\s*help\s*\))|(\bcitation needed\b)", flags=re.IGNORECASE)

    # Headings / list-y lines
    HEADING_LIKE = re.compile(r"^\s*(=+.+?=+)\s*$")
    LIST_LIKE = re.compile(r"^\s*([*#•\-–—]+)\s+")

    # Too many digits or weird symbol density (often IDs, coordinates, tables)
    MANY_DIGITS = re.compile(r"\d{4,}")  # years/IDs; used with ratios below
    NON_WORD_DENSITY = re.compile(r"[^\w\s]", flags=re.UNICODE)

    # Language-specific meta words (optional, small boost)
    LANG_META = {
        "de": [r"\bStand\b", r"\bAbruf\b", r"\bSeite\b"],
        "es": [r"\bConsultado\b", r"\bRecuperado\b", r"\bArchivado\b", r"\bEnlace\b"],
        "fr": [r"\bConsulté\b", r"\bArchivé\b", r"\bLien\b"],
        "ru": [r"\bдата обращения\b", r"\bархивировано\b", r"\bссылка\b", r"\bстраниц[аы]\b"],
    }
    lang_meta_patterns = LANG_META.get(lang_code, [])
    META_RE = re.compile("|".join(META_PATTERNS + lang_meta_patterns), flags=re.IGNORECASE)

    # Repeated punctuation / visual separators
    SEPARATORS = re.compile(r"([=_\-]{4,}|[•·]{3,}|\.{4,})")

    def is_junk_sentence(sent: str) -> bool:
        if not sent:
            return True

        # Normalize to reduce weird Unicode forms (keeps letters/digits)
        s = unicodedata.normalize("NFKC", sent).strip()

        # Very short lines are often captions, headers, leftovers
        # (You already filter min_words later, but this saves work earlier.)
        if len(s) < 15:
            return True

        # Common structural junk
        if HEADING_LIKE.match(s) or LIST_LIKE.match(s):
            return True

        # URL/HTML/wiki markup
        if URL_OR_HTML.search(s) or WIKI_MARKUP.search(s) or REF_LIKE.search(s):
            return True

        # Wikipedia/meta keywords
        if META_RE.search(s):
            return True

        # Visual separators
        if SEPARATORS.search(s):
            return True

        # Symbol/digit heuristics (conservative)
        # If sentence has many digits OR punctuation density is very high, reject.
        n_chars = len(s)
        n_digits = sum(ch.isdigit() for ch in s)
        n_punct = len(NON_WORD_DENSITY.findall(s))

        # digit ratio threshold; allows normal years but blocks coordinate/ID-heavy lines
        if n_digits / max(1, n_chars) > 0.18:
            return True

        # very long digit runs (IDs, coordinates, tables)
        if MANY_DIGITS.search(s):
            # only reject if also looks "non-sentence-y"
            if n_digits / max(1, n_chars) > 0.10:
                return True

        # punctuation density threshold; captions/refs often have lots of punctuation
        if n_punct / max(1, n_chars) > 0.25:
            return True

        return False

    return is_junk_sentence

class ColorPreprocessor:
    """Preprocessor for extracting and analyzing color adjectives in text."""
    
    def __init__(self, lang_code: str, data_dir: str = "data", verbose: bool = True):
        """
        Initialize preprocessor for a specific language.
        
        Args:
            lang_code: Language code (de, es, fr, ru)
            data_dir: Directory containing input/output data
            verbose: Whether to print progress messages
        """
        if lang_code not in LANG_CONFIG:
            raise ValueError(f"Unsupported language: {lang_code}. Supported: {list(LANG_CONFIG.keys())}")
        
        self.lang_code = lang_code
        self.data_dir = Path(data_dir)
        self.config = LANG_CONFIG[lang_code]
        self.verbose = verbose
        
        # Load spacy model
        if self.verbose:
            print(f"Loading spacy model: {self.config['spacy_model']}")
        self.nlp = spacy.load(self.config["spacy_model"])
        
        # Disable unused components for speed
        # Keep lemmatizer and parser (needed for dep_ and lemma_)
        # # Disable NER since we only check ent_type_ which is fast
        # if "ner" in self.nlp.pipe_names:
        #     self.nlp.disable_pipe("ner")
        
        # Get color words and normalize them
        self.colors = [c.lower() for c in self.config["colors"]]
        self.colors_set = frozenset(self.colors)  # frozenset is faster for lookups
        self.adj_dep_label = self.config.get("adj_dep_label", "amod")
        
        # Compile regex patterns
        color_pattern = r"\b(" + "|".join(re.escape(c) for c in self.colors) + r")\b"
        self.color_regex = re.compile(color_pattern, flags=re.IGNORECASE)
        # self.sentence_splitter = re.compile(r"(?<=[.!?])\s+")
        self.sentence_splitter = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")
        self.valid_sentence_regex = re.compile(self.config["valid_chars_regex"], re.UNICODE)
        self.junk_pattern = re.compile(r'TOK|\\N\d+|\t\d+\t')
        # check if sentence ends with punctuation (to filter out incomplete sentences)
        self.sentence_end_pattern = re.compile(r'[.!?]$')
        self.is_junk_sentence = build_is_junk_sentence(self.lang_code)


        # --- Wikipedia cleanup patterns (to reduce citations / markup / fragments) ---
        self._citation_pattern = re.compile(r"\s*\[(?:\d+|citation needed|note\s*\d+|nb\s*\d+|ref)\]\s*", re.IGNORECASE)
        # Generic short bracketed chunks like [12], [source], [réf.], etc.
        self._brackets_short_pattern = re.compile(r"\s*\[[^\]]{1,60}\]\s*")
        self._tags_pattern = re.compile(r"</?[^>]+>")
        self._template_pattern = re.compile(r"\{\{[^}]{1,200}\}\}")
        self._multispace_pattern = re.compile(r"\s+")
        self._space_before_punct_pattern = re.compile(r"\s+([,.;:!?])")
        self._control_pattern = re.compile(r"[\u0000-\u001F\u007F]")

        # Default sentence-length constraints (applied in extraction & filtering)
        self.default_min_words = 5
        self.default_max_words = 30

        # If no component provides sentence boundaries, add a lightweight sentencizer.
        # (Does not change your public API; used only if you choose to switch splitting strategy later.)
        if "parser" not in self.nlp.pipe_names and "senter" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def _is_color_adj_modifying_noun(self, token) -> bool:
        """Check if token is a color adjective modifying a noun."""
        if token.pos_ != "ADJ":
            return False
        # Check dependency label
        if isinstance(self.adj_dep_label, set):
            if token.dep_ not in self.adj_dep_label:
                return False
        elif token.dep_ != self.adj_dep_label:
            return False
        return True


    def _clean_wiki_sentence(self, s: str) -> str:
        """Remove common Wikipedia artifacts (citations, markup, odd whitespace)."""
        if not s:
            return ""
        s = s.replace("\u00a0", " ")
        s = self._control_pattern.sub(" ", s)
        s = self._tags_pattern.sub(" ", s)
        s = self._template_pattern.sub(" ", s)
        s = self._citation_pattern.sub(" ", s)
        s = self._brackets_short_pattern.sub(" ", s)
        s = self._space_before_punct_pattern.sub(r"\1", s)
        s = self._multispace_pattern.sub(" ", s).strip()
        return s

    @staticmethod
    def _has_verb(doc) -> bool:
        return any(t.pos_ in ("VERB", "AUX") for t in doc)

    @staticmethod
    def _alpha_word_count_from_doc(doc) -> int:
        return sum(1 for t in doc if t.is_alpha)

    def _extract_color_adj_pairs_from_doc(self, doc) -> list:
        """
        Extract (noun_lemma, adj_lemma) pairs from a spaCy doc.
        
        Returns list of pairs where adjective is a color and modifies a noun.
        """
        pairs = []
        for token in doc:
            if not self._is_color_adj_modifying_noun(token):
                continue
            adjective = token
            noun = token.head
            # Filter by colors
            if adjective.lemma_.lower() not in self.colors_set:
                continue
            # Skip named entities
            if adjective.ent_type_ or noun.ent_type_:
                continue
            # Only alphabetic nouns
            if not noun.text.isalpha():
                continue
            pairs.append((noun.lemma_, adjective.lemma_))
        return pairs

    def extract_color_sentences(self, input_file: str) -> list:
        """
        Extract sentences containing color words from Wikipedia dump.
        
        Args:
            input_file: Path to Wikipedia text file
            
        Returns:
            List of sentences containing color words
        """
        matched_sentences = []
        input_path = self.data_dir / input_file
        
        if self.verbose:
            print(f"Processing {input_path}...")
        
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f if not self.verbose else tqdm(f, desc="Extracting color sentences")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Split into sentences
                sentences = self.sentence_splitter.split(line)
                for sent in sentences:
                    sent = self._clean_wiki_sentence(sent.strip())
                    if not sent:
                        continue

                    # Enforce basic completeness: must end with sentence punctuation
                    if not self.sentence_end_pattern.search(sent):
                        continue

                    # Length filter (5–30 words by default)
                    wc = len(sent.split())
                    if wc < self.default_min_words or wc > self.default_max_words:
                        continue

                    # Filter common "fragment" patterns (often headings / field:value stubs)
                    if ":" in sent and wc < 10:
                        continue

                    # Check if sentence contains a color and has valid characters
                    if not (self.color_regex.search(sent) and self.valid_sentence_regex.match(sent)):
                        continue

                    # Skip junk sentences
                    if self.is_junk_sentence(sent):
                        continue
                    if self.junk_pattern.search(sent):
                        continue

                    matched_sentences.append(sent)
        
        if self.verbose:
            print(f"Found {len(matched_sentences)} sentences with colors.")
        return matched_sentences
    
    def extract_noun_adj_constructions(self, sentence: str, only_colors: bool = True) -> tuple:
        """
        Extract (noun, adjective) pairs where adjective modifies noun.
        Only returns pairs if sentence has exactly one such construction.
        
        Args:
            sentence: Input sentence
            only_colors: If True, only extract color adjectives
            
        Returns:
            Tuple of (pairs, sentences) where pairs is list of (noun_lemma, adj_lemma)
        """
        doc = self.nlp(sentence)
        pairs = self._extract_color_adj_pairs_from_doc(doc) if only_colors else []
        
        if not only_colors:
            # Non-color adjective extraction (original logic without color filter)
            for token in doc:
                if not self._is_color_adj_modifying_noun(token):
                    continue
                noun = token.head
                if token.ent_type_ or noun.ent_type_:
                    continue
                if not noun.text.isalpha():
                    continue
                pairs.append((noun.lemma_, token.lemma_))
        
        # Only keep sentences with exactly one noun-adj pair
        if len(pairs) == 1:
            return pairs, [sentence]
        
        return [], []
    
    def process_constructions(self, sentences: list, min_words: int = 5, max_words: int = 30) -> tuple:
        """
        Process sentences to extract noun-adjective constructions.
        
        Args:
            sentences: List of sentences
            min_words: Minimum sentence length in words
            max_words: Maximum sentence length in words
            
        Returns:
            Tuple of (constructions, filtered_sentences)
        """
        all_constructions = []
        all_sentences = []
        
        if self.verbose:
            print("Extracting noun-adjective constructions...")
        
        # Use nlp.pipe for batch processing (much faster)
        docs = self.nlp.pipe(sentences, batch_size=50)
        
        if self.verbose:
            docs = tqdm(docs, total=len(sentences), desc="Processing constructions")
        
        for sent, doc in zip(sentences, docs):
            pairs = self._extract_color_adj_pairs_from_doc(doc)
            
            # Only keep sentences with exactly one noun-adj pair
            if len(pairs) == 1:
                word_count = self._alpha_word_count_from_doc(doc)
                if not (min_words <= word_count <= max_words):
                    continue

                # Require at least one verb-like token to avoid title/list fragments
                if not self._has_verb(doc):
                    continue

                all_constructions.extend(pairs)
                all_sentences.append(sent)
        
        if self.verbose:
            print(f"Extracted {len(all_constructions)} constructions from {len(all_sentences)} sentences.")
        return all_constructions, all_sentences
    
    def generate_sentence_pairs(self, sentences: list, max_pairs: int = 10000) -> list:
        """
        Generate pairs of (sentence_with_adj, sentence_without_adj, adjective, noun).
        
        Args:
            sentences: List of sentences with color adjectives
            max_pairs: Maximum number of pairs to generate
            
        Returns:
            List of tuples (sentence_with_adj, sentence_without_adj, adjective, noun)
        """
        sentence_pairs = []
        
        if self.verbose:
            print("Generating sentence pairs...")
        
        # Process up to max_pairs * 2 to ensure we get enough valid pairs
        candidates = sentences[:max_pairs * 2]
        
        # Use nlp.pipe for batch processing
        docs = self.nlp.pipe(candidates, batch_size=50)
        
        if self.verbose:
            docs = tqdm(docs, total=len(candidates), desc="Creating pairs")
        
        for sent, doc in zip(candidates, docs):
            if len(sentence_pairs) >= max_pairs:
                break
            # Find color adjectives modifying nouns
            color_adjs = [
                token for token in doc 
                if self._is_color_adj_modifying_noun(token) 
                and token.lemma_.lower() in self.colors_set
            ]
            if len(color_adjs) != 1:
                continue
            adj = color_adjs[0]
            noun = adj.head
            # Create a sentence without the adjective
            sent_without_adj = "".join(token.text_with_ws for token in doc if token.i != adj.i).strip()
            # Normalize whitespace/spaces before punctuation
            sent_without_adj = self._space_before_punct_pattern.sub(r"\1", sent_without_adj)
            sent_without_adj = self._multispace_pattern.sub(" ", sent_without_adj)
            sentence_pairs.append((sent, sent_without_adj, adj.lemma_, noun.lemma_))
        
        if self.verbose:
            print(f"Generated {len(sentence_pairs)} sentence pairs.")
        return sentence_pairs
    
    def run_pipeline(self, input_file: str, output_prefix: str = None, max_pairs: int = 10000):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            input_file: Name of Wikipedia dump file (e.g., 'wikipedia_spanish_300k.txt')
            output_prefix: Prefix for output files (default: lang_code)
            max_pairs: Maximum number of sentence pairs to generate
        """
        if output_prefix is None:
            output_prefix = self.lang_code
        
        # Step 1: Extract color sentences
        if self.verbose:
            print(f"\n=== Step 1: Extracting color sentences ===")
        color_sentences = self.extract_color_sentences(input_file)
        
        # Step 2: Extract constructions
        if self.verbose:
            print(f"\n=== Step 2: Extracting noun-adjective constructions ===")
        constructions, filtered_sentences = self.process_constructions(color_sentences)
        
        # Step 3: Generate sentence pairs
        if self.verbose:
            print(f"\n=== Step 3: Generating sentence pairs ===")
        sentence_pairs = self.generate_sentence_pairs(filtered_sentences, max_pairs)
        
        # Step 4: Save outputs
        if self.verbose:
            print(f"\n=== Step 4: Saving outputs ===")
        self.save_outputs(
            constructions=constructions,
            sentences=filtered_sentences,
            sentence_pairs=sentence_pairs,
            output_prefix=output_prefix
        )
        
        # Step 5: Print statistics
        if self.verbose:
            self.print_statistics(constructions, filtered_sentences, sentence_pairs)
    
    def save_outputs(self, constructions: list, sentences: list, 
                     sentence_pairs: list, output_prefix: str):
        """Save all outputs to files."""
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Save sentence pairs
        pairs_file = output_dir / f"{output_prefix}_sentence_pairs_color_adj.pkl"
        with open(pairs_file, "wb") as f:
            pickle.dump(sentence_pairs, f)
        if self.verbose:
            print(f"Saved sentence pairs to {pairs_file}")
        
        # Save constructions
        constructions_file = output_dir / f"{output_prefix}_constructions.pkl"
        with open(constructions_file, "wb") as f:
            pickle.dump(constructions, f)
        if self.verbose:
            print(f"Saved constructions to {constructions_file}")
        
        # Save filtered sentences
        sentences_file = output_dir / f"{output_prefix}_filtered_sentences.txt"
        with open(sentences_file, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent + "\n")
        if self.verbose:
            print(f"Saved filtered sentences to {sentences_file}")
        
        # Save construction counts
        counts_file = output_dir / f"{output_prefix}_construction_counts.txt"
        construction_counts = Counter(constructions)
        with open(counts_file, "w", encoding="utf-8") as f:
            f.write("Noun-Color Adjective Counts (Top 50)\n")
            f.write("=" * 50 + "\n")
            for (noun, color), count in construction_counts.most_common(50):
                f.write(f"{noun} + {color}: {count}\n")
        if self.verbose:
            print(f"Saved construction counts to {counts_file}")
    
    def print_statistics(self, constructions: list, sentences: list, sentence_pairs: list):
        """Print statistics about the processed data."""
        print("\n" + "=" * 60)
        print(f"STATISTICS FOR {self.lang_code.upper()}")
        print("=" * 60)
        print(f"Total color sentences found: {len(sentences)}")
        print(f"Total noun-adjective constructions: {len(constructions)}")
        print(f"Total sentence pairs generated: {len(sentence_pairs)}")
        print(f"Unique constructions: {len(set(constructions))}")
        
        construction_counts = Counter(constructions)
        print(f"\nTop 10 most common constructions:")
        for (noun, color), count in construction_counts.most_common(10):
            print(f"  {noun} + {color}: {count}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia dumps to extract color adjective constructions"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all languages (de, es, fr, ru) automatically"
    )
    parser.add_argument(
        "--lang",
        choices=["de", "es", "fr", "ru"],
        help="Language code (required if not using --batch)"
    )
    parser.add_argument(
        "--input-file",
        help="Wikipedia dump filename (required if not using --batch)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input/output data (default: data)"
    )
    parser.add_argument(
        "--output-prefix",
        help="Prefix for output files (default: language code)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10000,
        help="Maximum number of sentence pairs to generate (default: 10000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with progress bars"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: process all languages
        batch_config = {
            "de": "wikipedia_german_300k.txt",
            "es": "wikipedia_spanish_300k.txt",
            "fr": "wikipedia_french_300k.txt",
            "ru": "wikipedia_russian_300k.txt"
        }
        
        data_path = Path(args.data_dir)
        print(f"Batch processing all languages...")
        print(f"Data directory: {args.data_dir}")
        print(f"Max pairs per language: {args.max_pairs}\n")
        
        results = {}
        for lang_code, input_file in batch_config.items():
            input_path = data_path / input_file
            
            if not input_path.exists():
                print(f"⚠ Skipping {lang_code}: {input_file} not found")
                results[lang_code] = "skipped"
                continue
            
            print(f"Processing {lang_code.upper()}... ", end="", flush=True)
            
            try:
                preprocessor = ColorPreprocessor(lang_code, args.data_dir, verbose=args.verbose)
                preprocessor.run_pipeline(input_file, output_prefix=lang_code, max_pairs=args.max_pairs)
                print("✓")
                results[lang_code] = "success"
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results[lang_code] = "error"
                continue
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        for lang, status in results.items():
            status_symbol = "✓" if status == "success" else "⚠" if status == "skipped" else "✗"
            print(f"{status_symbol} {lang}: {status}")
    else:
        # Single language mode
        if not args.lang or not args.input_file:
            parser.error("--lang and --input-file are required when not using --batch")
        
        preprocessor = ColorPreprocessor(args.lang, args.data_dir, verbose=args.verbose)
        preprocessor.run_pipeline(
            args.input_file,
            args.output_prefix,
            args.max_pairs
        )
        
        print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()
