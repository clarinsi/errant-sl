# SloERRANT

SloERRANT is a Slovenian adaptation of ERRANT (ERRor ANnotation Toolkit). It automatically extracts and classifies grammatical errors in Slovenian text.

Given an original sentence and its correction, SloERRANT finds the edits that turn one into the other and gives each edit an error type. It also adds a second level of annotation: edits that belong to the same morphological change, such as a case change spreading through a noun phrase, are linked into one **group** with a label such as `M-CASE`. The output uses the standard M2 format, and an evaluation script is included.

SloERRANT is based on the original [ERRANT](https://github.com/chrisjbryant/errant) toolkit developed by Bryant et al. English annotation (`-lang en`) still works as in upstream ERRANT v3.0.2.

```
S Govorili smo do tega odnosa .
A 2 3|||R:V-PREP|||o|||REQUIRED|||G0=R:M-GOV|||0
A 3 4|||R:M-PRON|||tem|||REQUIRED|||G0=R:M-GOV|||0
A 4 5|||R:M-NOUN|||odnosu|||REQUIRED|||G0=R:M-GOV|||0
```

## Differences from ERRANT

**Changes to the error taxonomy**
- Labels are **level-first** (O (orthography) / M (morphology) / V (vocabulary)/ S (syntax)).

**Slovene merger** (`errant/sl/merger.py`)
- The English merge rules are removed. Edits stay word-level, so the grouper can link them.
- These spans become a single edit:
  - a phrase ↔ a single pronoun (*te vere pa → katere*)
  - a multi-word ↔ a single connector (*za kar → zato*)
  - hyphen spacing (*e - pošto → e-pošto*)
- A rewritten clause becomes one `S-REWRITE` edit. A clause counts as rewritten when ≥ 3 words, making up > 40 % of the clause, change.
- A moved word or phrase becomes one `S-WO` edit.

**Slovene classifier** (`errant/sl/classifier.py`)
- Labels are `<operation>:<level>-<POS>`. As in ERRANT, the operation is `M` (missing), `U` (unnecessary) or `R` (replacement). The level is O, M, V or S. One-word replacements are decided from the tagger's lemmas and POS and the written forms, without a lexicon.
- **O (orthography)**
  - `O-SPELL`: a misspelling (*zeleli → želeli*, *srečau → srečal*, *z sestro → s sestro*)
  - `O-PUNCT`: punctuation
  - `O-CAP`: capitalisation
  - `O-MERGE`, `O-SPLIT`: spacing (*za njo → zanjo*)
- **M (morphology):** the same word in another form
  - `M-<POS>`: only the ending differs (*svojem → svojemu*)
  - bare `M`: the same lemma with a different POS
- **V (vocabulary):** a different word
  - `V-<POS>`: e.g. *iz → z*, *napravila → pripravila*, *ki → kateri*
  - bare `V`: the two words have a different POS
- **S (syntax)**
  - `S-<POS>`: a missing or unnecessary word
  - `S-WO`: word order
  - `S-REWRITE`: a rewritten phrase or clause
  - `S-REF`: a phrase ↔ a pronoun (*te vere pa → katere*)
  - `S-CONJ`: punctuation ↔ a conjunction
- `UNK`: detected but not corrected. `OTHER`: not covered by the rules.

**Optional grouper** (`errant/sl/grouper.py`)
- A new stage after the classifier. It links the edits of one morphological change into a group. Links come from agreement, from an auxiliary to its verb, and from a preposition to the case it governs.
- Each group is labelled by the feature that changed (`M-CASE`, `M-NUM`, `M-MOOD`, `M-GOV`, …). The label goes in the M2 comment field as `G<n>=<operation>:<label>`.
- Word labels are never changed. Skip the grouper with `annotate(..., group=False)`.

**Linguistic resources and NLP models**
- spaCy's Slovenian pipelines `sl_core_news_sm` (default) and `sl_core_news_trf` (recommended). Any other spaCy model can be selected with `-model`.
- **Pre-tagged CoNLL-U input** (`-format conllu`), so you can use CLASSLA, Trankit, UDPipe or any other tagger instead of spaCy.
- A table of the case/number readings of all personal and reflexive pronoun forms (long, short and bound: *njega, ga, zanj*), written from the Slovene pronoun paradigm. The grouper uses it for `M-PFORM`.

- **Scoring** (`errant_compare`) adds:
  - `-cat 4`: scores by level (O, M, V, S)
  - `-group`: scores morphology groups as single units
  
  Without these flags, the output is identical to upstream.

## Installation

You need Python ≥ 3.9. Install SloERRANT and the Slovenian spaCy model with a single command, preferably in a fresh virtual environment:

```
python3 -m venv sloerrant_env
source sloerrant_env/bin/activate
pip install "sloerrant[sl] @ git+https://github.com/zivastebljaj/errant-sl.git"
```

The `[sl]` extra installs `sl_core_news_sm` automatically, so you don't need a separate `spacy download` step. With [uv](https://docs.astral.sh/uv/), the same command runs faster: `uv pip install "sloerrant[sl] @ git+…"`.

**Recommended: the transformer model.** Lemmas decide between morphology (M) and vocabulary (V), and the small model's lemmas are much worse (see the examples below). To install the transformer model as well:

```
pip install "sloerrant[sl,trf] @ git+https://github.com/zivastebljaj/errant-sl.git"
```

This pulls in PyTorch, which is a large download. If you already tag your data with another parser, skip the model and use CoNLL-U input instead.

**From source** (for development):

```
git clone https://github.com/zivastebljaj/errant-sl.git
cd errant-sl
uv sync --extra sl          # or: pip install -e ".[sl]"
```

> **Note:** SloERRANT keeps the import name `errant`. Do not install it in the same environment as upstream `errant` from PyPI: the two would overwrite each other.

## Usage

### Python API

```python
import errant

annotator = errant.load("sl")                              # sl_core_news_sm
# annotator = errant.load("sl", model="sl_core_news_trf")  # recommended

orig = annotator.parse("Pomagal sem svojem dobrem prijatelju.", tokenise=True)
cor  = annotator.parse("Pomagal sem svojemu dobremu prijatelju.", tokenise=True)
for e in annotator.annotate(orig, cor):
    print(e.o_str, "->", e.c_str, e.type, e.group_label)
```

Output with `sl_core_news_trf`:

```
svojem -> svojemu R:M-PRON M-CASE
dobrem -> dobremu R:M-ADJ M-CASE
```

Each edit has the standard ERRANT attributes (`o_start`, `o_end`, `o_str`, `c_str`, `type`, …) plus `group`, `group_label` and `group_op`. Use `annotate(..., group=False)` to skip the grouper. `import_edit` classifies an edit but doesn't group it; call `annotator.grouper.group(orig, cor, edits)` for that.

**Pre-tagged CoNLL-U.** Only the vocabulary of the spaCy pipeline is used here, so a blank one is enough:

```python
import errant, spacy

annotator = errant.load("sl", nlp=spacy.blank("sl"))
for orig, cor in zip(annotator.read_conllu("orig.conllu"), annotator.read_conllu("cor.conllu")):
    edits = annotator.annotate(orig, cor)
```

### Command line

```
# Parallel text → M2 (one sentence per line)
errant_parallel -lang sl -tok -orig orig.txt -cor cor.txt [cor2.txt ...] -out out.m2
errant_parallel -lang sl -tok -model sl_core_news_trf -orig orig.txt -cor cor.txt -out out.m2

# Pre-tagged CoNLL-U → M2 (spaCy is not run)
errant_parallel -lang sl -format conllu -orig orig.conllu -cor cor.conllu -out out.m2

# Re-annotate an existing M2 file: -gold re-classifies its edits, -auto extracts new ones
errant_m2 -lang sl -gold in.m2 -out out.m2

# Evaluate a hypothesis against a reference
errant_compare -hyp hyp.m2 -ref ref.m2              # span-based correction, F0.5
errant_compare -hyp hyp.m2 -ref ref.m2 -cat 4       # per level: O, M, V, S
errant_compare -hyp hyp.m2 -ref ref.m2 -cat 3       # per label, e.g. R:M-NOUN
errant_compare -hyp hyp.m2 -ref ref.m2 -group -cat 3  # groups as units, e.g. M-CASE
```

- **Use `-tok` unless your text is already tokenised.** Without it, input is split on whitespace only, so end of sentence punctuation stays glued to the previous token.
- **CoNLL-U:** sentences are paired by their order in each file. Multi-word tokens are split into syntactic words (*vanj* → *v nj*), and M2 offsets refer to those words.
- **`-group`:** a group is a true positive only if the other file has a group with exactly the same member edits. Both files must have been annotated by SloERRANT.
- Annotate the hypothesis and the reference with the **same model or tagger**. Mixing parsers lowers scores slightly.
- All commands accept `-h` for further options.

### Example

`orig.txt`:
```
To je zelo zanimva knjiga.
Pomagal sem svojem dobrem prijatelju.
Govorili smo do tega odnosa.
Kot da dela.
Sinoči sva šla z sestro v kino.
Gre dvakrat vsak teden na trening.
```
`cor.txt`:
```
To je zelo zanimiva knjiga.
Pomagal sem svojemu dobremu prijatelju.
Govorili smo o tem odnosu.
Kot da bi delala.
Sinoči sva šla s sestro v kino.
Gre vsak teden dvakrat na trening.
```

`errant_parallel -lang sl -tok -model sl_core_news_trf -orig orig.txt -cor cor.txt -out out.m2` gives:

```
S To je zelo zanimva knjiga .
A 3 4|||R:O-SPELL|||zanimiva|||REQUIRED|||-NONE-|||0

S Pomagal sem svojem dobrem prijatelju .
A 2 3|||R:M-PRON|||svojemu|||REQUIRED|||G0=R:M-CASE|||0
A 3 4|||R:M-ADJ|||dobremu|||REQUIRED|||G0=R:M-CASE|||0

S Govorili smo do tega odnosa .
A 2 3|||R:V-PREP|||o|||REQUIRED|||G0=R:M-GOV|||0
A 3 4|||R:M-PRON|||tem|||REQUIRED|||G0=R:M-GOV|||0
A 4 5|||R:M-NOUN|||odnosu|||REQUIRED|||G0=R:M-GOV|||0

S Kot da dela .
A 2 2|||M:S-AUX|||bi|||REQUIRED|||G0=R:M-MOOD|||0
A 2 3|||R:M-VERB|||delala|||REQUIRED|||G0=R:M-MOOD|||0

S Sinoči sva šla z sestro v kino .
A 3 4|||R:O-SPELL|||s|||REQUIRED|||-NONE-|||0

S Gre dvakrat vsak teden na trening .
A 1 4|||R:S-WO|||vsak teden dvakrat|||REQUIRED|||-NONE-|||0
```

These are the examples of:
1. A misspelling.
2. A case error spread over two agreeing words, which form one `M-CASE` group.
3. A wrong preposition together with the case it governs, labelled `M-GOV`.
4. A missing conditional *bi* and the verb form it requires, labelled `M-MOOD`.
5. The *s/z* preposition confusion.
6. A word order change.

With the default `sl_core_news_sm`, sentences 2 and 4 come out worse. *dobremu* and *delala* are labelled `R:O-SPELL`, so neither joins its group, because the small model gives them the wrong lemmas.

## Error Categories

Each edit has a **word label** `<operation>:<label>`. As in ERRANT, the operation is `M` (missing), `U` (unnecessary) or `R` (replacement). Unlike ERRANT, the label starts with a **level**:

| Level | Labels | Meaning |
|---|---|---|
| **O**: orthography | `O-SPELL`, `O-PUNCT`, `O-CAP`, `O-MERGE` (*za njo → zanjo*), `O-SPLIT` (*črnobelo → črno - belo*) | spelling, punctuation, capitalisation, spacing |
| **M**: morphology | `M-<POS>`, `M` | same word (same or close lemma), different form; bare `M` = same lemma, different POS |
| **V**: vocabulary | `V-<POS>`, `V` | a different word; bare `V` = the two sides have a different POS |
| **S**: syntax | `S-<POS>`, `S-WO`, `S-REWRITE`, `S-REF`, `S-CONJ` | a missing or unnecessary word (`S-<POS>`), word order, a rewritten clause or phrase, phrase ↔ pronoun (*te vere pa → katere*), punctuation ↔ conjunction |
| | `OTHER`, `UNK` | unclassified; detected but not corrected |

`<POS>` is the Universal Dependencies UPOS tag, with these renames: `ADP` → `PREP`, `PROPN` → `NOUN`, `DET` and the relative *ki* → `PRON`, `CCONJ` / `SCONJ` → `CONJ`. A token without letters or digits is `PUNCT`. A missing or unnecessary auxiliary is `S-AUX`; its morphological effect is recorded in the group.

**Group labels** (morphology only) appear in the M2 comment field as `G<n>=<operation>:<label>`:

| Label | Change |
|---|---|
| `M-CASE`, `M-NUM`, `M-GEND`, `M-PRS` | case, number, gender, person |
| `M-GOV` | case change together with a preposition change (*do tega odnosa → o tem odnosu*) |
| `M-DEF`, `M-DEG` | definiteness, degree of comparison (incl. *bolj* / *najbolj*) |
| `M-TENSE`, `M-ASP`, `M-MOOD` | tense, aspect, mood (incl. conditional *bi*) |
| `M-VFORM`, `M-SFORM` | non-finite ↔ finite verb; infinitive ↔ supine (*skrbet → skrbeti*) |
| `M-PFORM` | long / short / bound pronoun form (*njemu → mu*, *za njega → zanj*) |
| `M-REFL` | reflexive *se* ↔ *si* |
| `M-MORPH` | other morphological change |

When several features change, the label follows this priority: `SFORM > MOOD > VFORM > TENSE > ASP > REFL > DEG > DEF > GOV > PFORM > CASE > PRS > NUM > GEND > MORPH`. Every morphology edit belongs to a group, even if it is a group of one. Prepositions, *se/si*, *bolj/najbolj* and auxiliaries can also join a group.

**Differences from English ERRANT**
- English ERRANT types are POS-first (`VERB:SVA`, `NOUN:NUM`, `VERB:TENSE`). SloERRANT puts the level first and keeps word labels coarse. The *kind* of morphological error, such as agreement, case or tense, moves to the group level, where it can cover several words.
- There are no `SVA`, `NOUN:INFL`, `VERB:FORM`, `ORTH` or `CONTR` types. Their roles are taken by group labels, `O-CAP` / `O-MERGE` / `O-SPLIT` and `O-SPELL`.
- New types: `S-REWRITE`, `S-REF`, `S-CONJ`, `S-WO` for moves over any distance, and bare `M` / `V`.

**Known limitations**
- Group types and the choice between M and V depend on how the tagger reads *incorrect* word forms. Use a good tagger.
- A word moved together with another change inside the moved span is not recognised as a move.
- Corrections that only change spacing, such as `20%` → `20 %`, are invisible at the token level.
- `errant_m2` re-parses with spaCy and has no CoNLL-U input.

## Citation

There is no SloERRANT publication yet. Please cite the software for now:

```bibtex
@software{stebljaj2026sloerrant,
  author = {Stebljaj, Živa},
  title  = {{SloERRANT}: A Slovenian Adaptation of the ERRor ANnotation Toolkit},
  year   = {2026},
  url    = {https://github.com/zivastebljaj/errant-sl}
}
```
<!-- TODO: replace with the SloERRANT paper once published -->

Please also cite the original ERRANT papers:

```bibtex
@inproceedings{bryant-etal-2017-automatic,
  title     = {Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction},
  author    = {Bryant, Christopher and Felice, Mariano and Briscoe, Ted},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/P17-1074/},
  pages     = {793--805}
}

@inproceedings{felice-etal-2016-automatic,
  title     = {Automatic Extraction of Learner Errors in {ESL} Sentences Using Linguistically Enhanced Alignments},
  author    = {Felice, Mariano and Bryant, Christopher and Briscoe, Ted},
  booktitle = {Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  year      = {2016},
  address   = {Osaka, Japan},
  url       = {https://aclanthology.org/C16-1079/},
  pages     = {825--835}
}
```

## Acknowledgements

SloERRANT is based on ERRANT: https://github.com/chrisjbryant/errant

The original ERRANT software was developed by Christopher Bryant and collaborators. For the full API reference (Alignment and Edit objects, merging strategies, etc.), see the [upstream README](https://github.com/chrisjbryant/errant#readme). The API is unchanged.

The Slovenian pipelines are [spaCy](https://spacy.io/models/sl)'s `sl_core_news` models by Explosion.

## License

MIT License. See [LICENSE.md](LICENSE.md). The original copyright of Christopher Bryant and Mariano Felice is retained.

SloERRANT doesn't ship any third-party data. The spaCy models are installed as dependencies under their own licenses.

Bug reports and suggestions: [GitHub issues](https://github.com/zivastebljaj/errant-sl/issues).
