from re import sub
from rapidfuzz.distance import Levenshtein
import spacy.symbols as POS
from errant.sl.merger import is_conn, is_ref

# Classifier resources
# Open class coarse Spacy POS tags
open_pos1 = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
# Label POS that inflect; only these can have morphology errors in the
# tagger-only rules (prepositions, conjunctions, particles don't inflect)
inflecting_pos = {"NOUN", "ADJ", "PRON", "VERB", "AUX", "NUM"}
# Closed-class label POS: the tagger knows all their members, so a change
# to another member is a different word, not a misspelling
closed_pos = {"PREP", "CONJ", "PART", "PRON"}
# Letters with a caron and their plain counterparts
strip_caron = str.maketrans("čšž", "csz")
# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "NUM", "SYM", "X"}
# Universal Dependencies relations that correspond to a coarse POS-like category
dep_map = {
    "amod": "ADJ",
    "advmod": "ADV",
    "det": "PRON",
    "case": "PREP",
    "punct": "PUNCT"}
# Single-letter prepositions confusable by voicing/pronunciation
prep_confusions = {frozenset({"s", "z"}), frozenset({"k", "h"}), frozenset({"u", "v"})}
# Lemma similarity above which two lemmas count as "close enough" for M-POS,
# tolerating lemmatizer noise instead of requiring an exact match
LEMMA_SIM_THRESHOLD = 0.8
# Divisor for scaling the allowed surface-text edit distance with word
# length (allowed = max(1, len // TEXT_EDIT_SCALE)), so short words aren't
# held to the same absolute distance as long ones
TEXT_EDIT_SCALE = 3

# Input 1: A resolved POS string
# Input 2: The level prefix: S (one-sided), M or V
# Output: The category label: O-PUNCT for punctuation, else <level>-<POS>
def label_pos(pos, level="S"):
    return "O-PUNCT" if pos == "PUNCT" else level + "-" + pos

# Input 1: A string
# Input 2: A string
# Output: Boolean; edit distance small relative to word length. Used for
# spelling, where the actual written characters are what matters.
def close_text(a, b):
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    allowed = max(1, max(len(a), len(b)) // TEXT_EDIT_SCALE)
    return Levenshtein.distance(a, b) <= allowed

# Input 1: Spacy orig token
# Input 2: Spacy cor token
# Output: Boolean; the lemmas are identical or similar enough to count as
# the same word. Used for morphology, where the abstract lemma matters, not
# the surface spelling; tolerates lemmatizer noise rather than requiring an
# exact match.
def close_lemma(o_tok, c_tok):
    if same_lemma(o_tok, c_tok):
        return True
    return Levenshtein.normalized_similarity(o_tok.lemma_.lower(), c_tok.lemma_.lower()) > LEMMA_SIM_THRESHOLD

# Input 1: Spacy orig token
# Input 2: Spacy cor token
# Output: Boolean; the lemmas are identical, ignoring case
def same_lemma(o_tok, c_tok):
    return o_tok.lemma_.lower() == c_tok.lemma_.lower()

# Input 1: The orig label POS
# Input 2: The cor label POS
# Output: The vocabulary label: V-<POS> if both sides have the same POS,
# V otherwise (O-PUNCT for punctuation)
def label_vocab(o_pos, c_pos):
    if c_pos == "PUNCT": return "O-PUNCT"
    return "V-" + c_pos if o_pos == c_pos else "V"

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Nothing to nothing is a detected but not corrected edit
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    # Missing
    elif not edit.o_toks and edit.c_toks:
        op = "M:"
        cat = get_one_sided_type(edit.c_toks)
        edit.type = op+cat
    # Unnecessary
    elif edit.o_toks and not edit.c_toks:
        op = "U:"
        cat = get_one_sided_type(edit.o_toks)
        edit.type = op+cat
    # Replacement and special cases
    else:
        # Same to same is a detected but not corrected edit
        if edit.o_str == edit.c_str:
            edit.type = "UNK"
        # Special: Ignore case change at the end of multi token edits
        # E.g. [Doctor -> The doctor], [, since -> . Since]
        # Classify the edit as if the last token wasn't there
        elif edit.o_toks[-1].lower == edit.c_toks[-1].lower and \
                (len(edit.o_toks) > 1 or len(edit.c_toks) > 1):
            # Store a copy of the full orig and cor toks
            all_o_toks = edit.o_toks[:]
            all_c_toks = edit.c_toks[:]
            # Truncate the instance toks for classification
            edit.o_toks = edit.o_toks[:-1]
            edit.c_toks = edit.c_toks[:-1]
            # Classify the truncated edit
            edit = classify(edit)
            # Restore the full orig and cor toks
            edit.o_toks = all_o_toks
            edit.c_toks = all_c_toks
        # Replacement
        else:
            op = "R:"
            cat = get_two_sided_type(edit.o_toks, edit.c_toks)
            edit.type = op+cat
    return edit

# Input: A Spacy token
# Output: The label POS of the token. A token without letters or digits is
# punctuation. UD POS renamed for readability (as upstream ERRANT), plus
# Slovene grammar: determiners and the relative ki
# are pronouns (zaimki), coordinating and subordinating conjunctions are one
# class (vezniki)
def label_pos_of(tok):
    if not any(ch.isalnum() for ch in tok.text): return "PUNCT"
    if tok.lower_ == "ki": return "PRON"
    if tok.pos_ == "ADP": return "PREP"
    if tok.pos_ == "PROPN": return "NOUN"
    if tok.pos_ == "DET": return "PRON"
    if tok.pos_ in {"CCONJ", "SCONJ"}: return "CONJ"
    return tok.pos_

# Input: Spacy tokens
# Output: A list of pos and dep tag strings
def get_edit_info(toks):
    pos = []
    dep = []
    for tok in toks:
        pos.append(label_pos_of(tok))
        dep.append(tok.dep_)
    return pos, dep

# Input: Spacy tokens
# Output: An error type string based on input tokens from orig or cor
# When one side of the edit is null, we can only use the other side
def get_one_sided_type(toks):
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # Punctuation next to a word belongs to it: [, da], [To ,]
    if set(pos_list) != {"PUNCT"}:
        dep_list = [d for p, d in zip(pos_list, dep_list) if p != "PUNCT"]
        pos_list = [p for p in pos_list if p != "PUNCT"]
    # POS-based tags. Ignores rare, uninformative categories; a missing /
    # unnecessary numeral is informative: [eden razlog -> razlog]
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_pos - {"NUM"}:
        return label_pos(pos_list[0])
    # More POS-based tags using special dependency labels
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return label_pos(dep_map[dep_list[0]])
    # Verb-related non-finite constructions
    if set(pos_list) == {"PART", "VERB"}:
        return label_pos("VERB")
    # A missing / unnecessary phrase of mixed POS is a structural change
    if len(pos_list) > 1 and "PUNCT" not in pos_list:
        return "S-REWRITE"
    # Tricky cases
    return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: An error type string based on orig AND cor
def get_two_sided_type(o_toks, c_toks):
    # Extract pos tags and parse info from the toks as lists
    o_pos, o_dep = get_edit_info(o_toks)
    c_pos, c_dep = get_edit_info(c_toks)

    # Orthography; i.e. whitespace and/or case errors.
    if only_orth_change(o_toks, c_toks):
        return get_orth_subtype(o_toks, c_toks)
    # Spacing around hyphens / dashes 
    if only_hyphen_change(o_toks, c_toks):
        if len(o_toks) == len(c_toks): return "O-PUNCT"
        return "O-MERGE" if len(o_toks) > len(c_toks) else "O-SPLIT"
    # Word Order; only matches exact reordering.
    if exact_reordering(o_toks, c_toks):
        return "S-WO"
    # Single-letter preposition confusion; both sides are real words, so
    # the word-level rules below can't catch these.
    if is_prep_confusion(o_toks, c_toks):
        return "O-SPELL"
    # Units the merger keeps together (phrase <-> pronoun)
    span_type = get_span_type(o_toks, c_toks)
    if span_type:
        return span_type

    # 1:1 replacements (very common)
    if len(o_toks) == len(c_toks) == 1:
        # Punctuation <-> conjunction: how clauses are linked, either
        # direction: [, -> in], [in -> ,]
        if {o_pos[0], c_pos[0]} == {"PUNCT", "CONJ"}:
            return "S-CONJ"
        # Morphology, vocabulary or spelling, from the tagger and the forms
        return get_word_type(o_toks[0], c_toks[0], o_pos[0], c_pos[0], o_dep[0], c_dep[0])

    # Multi-token replacements that are not orthography, word order, REF or
    # connector spans (all handled above): rewritten phrases / clauses
    return "S-REWRITE"

# Input 1-2: Spacy orig and cor token of a 1:1 replacement
# Input 3-4: Their label POS
# Input 5-6: Their dependency labels
# Output: The error type (morphology, vocabulary or spelling) from the
# tagger's lemmas and POS and the written forms
def get_word_type(o_tok, c_tok, o_pos, c_pos, o_dep, c_dep):
    # The orig lemma is evidence only if the tagger lemmatised the form; an
    # unlemmatised form (lemma = form) leaves the decision to the ending test
    # [otroci/otroci -> otroki/otrok, tistemu/tistemu -> tistem/tisti]
    lemma_match = close_lemma(o_tok, c_tok) or o_tok.lemma_.lower() == o_tok.lower_
    # Pronunciation spelling of the l-participle: final -l written as -u / -v
    # / -o, often with the -e- of -el dropped [srečau -> srečal, vedu -> vedel]
    if is_l_participle_spelling(o_tok.lower_, c_tok):
        return "O-SPELL"
    # 1. MORPHOLOGY: the same word (close lemma, same inflecting POS), only
    # the ending differs [svojem -> svojemu, mati -> mater]
    if lemma_match and o_pos == c_pos and o_pos in inflecting_pos and \
            o_tok.text.isalpha() and c_tok.text.isalpha() and \
            ending_only(o_tok.lower_, c_tok.lower_, c_tok.lemma_.lower()):
        return label_pos(o_pos, "M")
    # Same lemma but the POS differs: morphology without a POS
    if same_lemma(o_tok, c_tok) and "PUNCT" not in {o_pos, c_pos}:
        return "M"
    # Auxiliaries with different lemmas
    if o_dep.startswith("aux") and c_dep.startswith("aux"):
        return "M-AUX"
    # 2. VOCABULARY: the tagger recognised the orig word as another word: a
    # different lemma, and the orig is a closed-class word or was lemmatised
    # (lemma != form) [mogla/moči -> morala/morati, iz -> z, kaj -> kar,
    # odvrne/odvrniti -> zavrne/zavrniti]
    if not close_lemma(o_tok, c_tok) and o_pos in closed_pos:
        return label_vocab(o_pos, c_pos)
    # A different start is a different word (a prefix) [napravila ->
    # pripravila, zapazimo -> opazimo], except u / v, which is spelling
    # [ustop -> vstop]
    if prefix_change(o_tok.lower_, c_tok.lower_) and not same_lemma(o_tok, c_tok):
        return label_vocab(o_pos, c_pos)
    # 3. SPELLING: the change is not in the ending, and the written forms are
    # close [zeleli -> želeli, odihov -> oddihov]
    if close_text(o_tok.text, c_tok.text):
        return "O-SPELL"
    # 4. VOCABULARY: anything else is a different word
    return label_vocab(o_pos, c_pos)

# Input 1: The orig form, lowercased
# Input 2: The cor form, lowercased
# Input 3: The cor lemma, lowercased
# Output: Boolean; the two forms differ only in the ending: the differing
# part reaches the end of the word (a shared last letter is allowed:
# [otroki -> otroci]) and starts no earlier than the stem of the cor form,
# i.e. its common prefix with the cor lemma. When the cor form has an ending
# of its own, one letter less is allowed for alternations and mobile vowels
# [otroki -> otroci, pes -> psa]; a cor form that is its bare stem gets no
# allowance, so a changed final consonant is spelling [obras -> obraz]
def ending_only(o, c, c_lemma):
    p = common_prefix(o, c)
    s = common_prefix(o[p:][::-1], c[p:][::-1])
    stem = common_prefix(c, c_lemma)
    allowance = 1 if len(c) > stem else 0
    return s <= 1 and p >= stem - allowance

# Input 1: The orig form, lowercased
# Input 2: The cor token
# Output: Boolean; the cor token is an l-participle (-l) and the orig form
# writes its final -l as -u / -v / -o, with or without the -e- of -el
# [srečau -> srečal, spremljav -> spremljal, vedu -> vedel]
def is_l_participle_spelling(o, c_tok):
    c = c_tok.lower_
    if "Part" not in c_tok.morph.get("VerbForm") or not c.endswith("l"):
        return False
    stems = {c[:-1]}
    if c.endswith("el"): stems.add(c[:-2])
    return any(o == stem + end for stem in stems for end in "uvo")

# Input 1: The orig form, lowercased
# Input 2: The cor form, lowercased
# Output: Boolean; the first letters differ in a way that is not a typical
# misspelling: not u / v [ustop -> vstop], not a missing caron [špolno ->
# spolno] and not two vowels [usnova -> osnova]
def prefix_change(o, c):
    if not o or not c or o[0] == c[0]:
        return False
    a, b = o[0].translate(strip_caron), c[0].translate(strip_caron)
    return a != b and {a, b} != {"u", "v"} and not {a, b} <= set("aeiou")

# Input 1-2: Two strings
# Output: The length of their common prefix
def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return n

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: The label of a non-decomposable unit the merger keeps as one edit,
# or None. 
def get_span_type(o_toks, c_toks):
    # ki <-> kateri: a choice between two equivalent relative pronouns, i.e.
    # vocabulary: [kateri -> ki], [kateremu -> ki mu], [ki -> s katerim]
    if is_ki_kateri(o_toks, c_toks) or is_ki_kateri(c_toks, o_toks):
        return "V-PRON"
    # A phrase replaced by a single pronoun, or the reverse: [te vere pa ->
    # katere], [kateremu -> ki mu].
    for a, b in ((o_toks, c_toks), (c_toks, o_toks)):
        if is_ref(a, b):
            if "Bound" in a[0].morph.get("Variant") and \
                    any(tok.pos in {POS.PRON, POS.DET} for tok in b):
                return "M-PRON"
            return "S-REF"
    # A multi-word connector replaced by a single one, or the reverse:
    # [za kar -> zato], [ker -> kljub temu da]. The relative ki is a
    # pronoun in Slovene grammar: [ki -> s katerim] is a reference change
    for a, b in ((o_toks, c_toks), (c_toks, o_toks)):
        if is_conn(a, b):
            return "S-REF" if a[0].lower_ == "ki" else "V-CONJ"
    return None

# Input 1: Tokens on one side of an edit
# Input 2: Tokens on the other side
# Output: True if a has the relative ki and b a form of kateri (matched on
# the form, kater-, since the lemmatiser often leaves these unlemmatised),
# and both sides are only function words: [ki -> s katerim], not [katere
# imenuju -> ki jih imenujejo]
def is_ki_kateri(a, b):
    if any(tok.pos not in {POS.PRON, POS.DET, POS.ADP, POS.SCONJ, POS.PUNCT}
            for tok in list(a)+list(b)):
        return False
    return any(tok.lower_ == "ki" for tok in a) and \
        not any(tok.lower_.startswith("kater") for tok in a) and \
        any(tok.lower_.startswith("kater") for tok in b)

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the difference between orig and cor is only whitespace or case
def only_orth_change(o_toks, c_toks):
    o_join = "".join([o.lower_ for o in o_toks])
    c_join = "".join([c.lower_ for c in c_toks])
    if o_join == c_join:
        return True
    return False

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; orig and cor are the same apart from spacing, case,
# hyphens, dashes and apostrophes
def only_hyphen_change(o_toks, c_toks):
    o_join = sub("['\\-–—]", "", "".join([o.lower_ for o in o_toks]))
    c_join = sub("['\\-–—]", "", "".join([c.lower_ for c in c_toks]))
    return o_join == c_join

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: The orthography subtype string; only_orth_change() must be true
def get_orth_subtype(o_toks, c_toks):
    o_exact = "".join(o.text for o in o_toks)
    c_exact = "".join(c.text for c in c_toks)
    if o_exact == c_exact:
        return "O-MERGE" if len(o_toks) > len(c_toks) else "O-SPLIT"
    return "O-CAP"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the tokens are exactly the same but in a different order
def exact_reordering(o_toks, c_toks):
    # Sorting lets us keep duplicates.
    o_set = sorted([o.lower_ for o in o_toks])
    c_set = sorted([c.lower_ for c in c_toks])
    if o_set == c_set:
        return True
    return False

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; both sides are single-letter s/z, k/h or u/v prepositions
def is_prep_confusion(o_toks, c_toks):
    if len(o_toks) == 1 and len(c_toks) == 1:
        o_low, c_low = o_toks[0].text.lower(), c_toks[0].text.lower()
        if len(o_low) == 1 and len(c_low) == 1:
            return frozenset({o_low, c_low}) in prep_confusions
    return False
