# Are pre-nominal color adjectives more efficient?

## Project Structure

```
├── preprocess_colors.py          # Step 1: Extract sentences with color adjectives
├── add_color_adjectives.py       # Step 2: Extract non-adj sentences & add inflected adjectives
├── analysis_removing_adj.ipynb   # Step 3: Main analysis notebook (UID computation)
├── setup.sh                      # Virtual environment setup script
├── requirements.txt              # Python dependencies
├── data/
│   ├── wikipedia_german_300k.txt
│   ├── wikipedia_spanish_300k.txt
│   ├── wikipedia_french_300k.txt
│   ├── wikipedia_russian_300k.txt
│   └── processed/
│       ├── {lang}_sentence_pairs_color_adj.pkl  # Sentence pairs (with/without adj)
│       ├── {lang}_constructions.pkl             # Noun-adjective constructions
│       ├── {lang}_filtered_sentences.txt        # Filtered sentences
│       └── {lang}_construction_counts.txt       # Construction statistics
└── figures/                      # Output figures (Figure1.pdf, Figure2.pdf)
```

## Setup

### 1. Create Virtual Environment

```bash
bash setup.sh
source .venv_color_adj/bin/activate
```

### 2. Download spaCy Language Models

```bash
python -m spacy download de_core_news_lg
python -m spacy download es_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download ru_core_news_lg
```

## Data Preparation

### Downloading Wikipedia Data

Download Wikipedia text dumps using [wikisets](https://github.com/sujaltv/wikisets):

```bash
pip install wikisets
```

```python
from wikisets import Wikiset, WikisetConfig

# Create config for 300k samples per language
config = WikisetConfig(
    languages=[
        {"lang": "de", "size": 300000},
        {"lang": "es", "size": 300000},
        {"lang": "fr", "size": 300000},
        {"lang": "ru", "size": 300000},
    ],
    seed=42
)

dataset = Wikiset(config)

# Save each language to separate files
for lang in ["de", "es", "fr", "ru"]:
    lang_names = {"de": "german", "es": "spanish", "fr": "french", "ru": "russian"}
    output_file = f"data/wikipedia_{lang_names[lang]}_300k.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for article in dataset.filter(lang=lang):
            text = article["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
    print(f"Saved {lang} to {output_file}")
```

Required files in `data/`:
- `wikipedia_german_300k.txt`
- `wikipedia_spanish_300k.txt`
- `wikipedia_french_300k.txt`
- `wikipedia_russian_300k.txt`

## Pipeline

### Step 1: Extract Sentences with Color Adjectives

Extract sentence pairs (with/without color adjective) from Wikipedia using spaCy:

```bash
python preprocess_colors.py --batch --verbose
```

This creates in `data/processed/`:
- `{lang}_sentence_pairs_color_adj.pkl` - Sentence pairs (with_adj, without_adj, adj_lemma, noun_lemma)
- `{lang}_constructions.pkl` - All extracted noun-adjective constructions
- `{lang}_filtered_sentences.txt` - Filtered sentences with color adjectives
- `{lang}_construction_counts.txt` - Top 50 most common constructions

### Step 2: Add Color Adjectives to Non-Adjective Sentences

Find sentences containing target nouns without adjectives, insert adjective placeholders, and use an LLM to inflect them:

```bash
python add_color_adjectives.py --n-samples 400
```

Options:
- `--n-samples N` - Number of sentence pairs per language (default: 400)
- `--model MODEL_ID` - HuggingFace model for inflection (default: Qwen/Qwen2.5-3B-Instruct)
- `--data-dir DIR` - Data directory (default: data)
- `--no-gpu` - Force CPU-only inference
- `--output FILE` - Output file (default: sentence_pairs_color_adj_added.pkl)

Output: `sentence_pairs_color_adj_added.pkl`

**Note:** This step requires downloading the Qwen model (~6GB).

### Step 3: Main Analysis (analysis_removing_adj.ipynb)

Compute UID (Uniform Information Density) scores:

1. Open `analysis_removing_adj.ipynb`
2. Load the mGPT model (`ai-forever/mGPT`, ~3GB download on first run)
3. Compute UID⁻¹ for sentence pairs with/without adjectives
4. Generate comparison histograms and statistical tests

Key outputs:
- `figures/Figure1.pdf` - UID difference distributions (removing adjectives)
- `figures/Figure2.pdf` - UID difference distributions (adding adjectives)
- Pairwise language comparison heatmap
