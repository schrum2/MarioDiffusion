
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence

# --- Generic English quantity vocabulary (domain independent, not game topics) ---

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}

# Single-word quantifiers / extent words.
SINGLE_QUANTIFIERS = {
    "few", "several", "many", "some", "numerous", "multiple", "various",
    "couple", "ton", "lot", "handful", "plenty",
    "full", "no", "all", "half", "mostly", "most",
}

# Multi-word quantifiers, replaced before tokenizing (longest first).
MULTI_QUANTIFIERS = [
    "a couple of", "a number of", "a handful of", "a ton of", "a lot of",
    "a few", "lots of", "plenty of",
]

STOPWORDS = {"a", "an", "the", "of", "with", "and", "or"}

QUANTITY_PLACEHOLDER = "<Q>"


def singularize(word: str) -> str:
    """Best-effort singularization of a single word (no game knowledge)."""
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


def normalize_phrase(phrase: str) -> str:
    """Lowercase, strip, and singularize each word of a phrase.

    Used to compare captions for equality independent of pluralization (so that
    "two coins" and "two coin" are treated as the same phrase).
    """
    words = phrase.lower().strip().split()
    return " ".join(singularize(w) for w in words)


def templatize(phrase: str) -> str:
    """Reduce a phrase to a topic template by abstracting away quantities."""
    padded = " " + phrase.lower().strip() + " "
    for multi in sorted(MULTI_QUANTIFIERS, key=len, reverse=True):
        padded = padded.replace(" " + multi + " ", " " + QUANTITY_PLACEHOLDER + " ")

    tokens = []
    for tok in padded.split():
        if (
            tok == QUANTITY_PLACEHOLDER
            or tok in NUMBER_WORDS
            or tok in SINGLE_QUANTIFIERS
            or re.fullmatch(r"[0-9]+", tok)
        ):
            tokens.append(QUANTITY_PLACEHOLDER)
        else:
            tokens.append(singularize(tok))

    # Collapse runs of placeholders ("a ton of" -> single <Q>).
    collapsed = []
    for tok in tokens:
        if tok == QUANTITY_PLACEHOLDER and collapsed and collapsed[-1] == QUANTITY_PLACEHOLDER:
            continue
        collapsed.append(tok)
    return " ".join(collapsed)


def _content_words(template: str) -> set:
    """Meaningful (non-quantity, non-stopword) words of a template."""
    return {t for t in template.split() if t != QUANTITY_PLACEHOLDER and t not in STOPWORDS}


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


class GenericGrammarGenerator:
    """Learns topics from a caption corpus and samples random captions from them."""

    def __init__(self, captions: Sequence[str], seed: int = 512, describe_absence: bool = False):
        if describe_absence:
            # Absence descriptions require knowing the full topic universe up front,
            # which is exactly what the per-game generators encode by hand. The
            # generic generator only sees topics that actually occur in the data,
            # so it cannot reliably describe what is absent.
            raise ValueError("GenericGrammarGenerator does not support describe_absence")

        random.seed(seed)
        self.describe_absence = describe_absence

        # template -> set of real surface phrases
        template_to_phrases: Dict[str, set] = defaultdict(set)
        # template set for every caption (used for co-occurrence and length stats)
        caption_template_sets: List[frozenset] = []

        for caption in captions:
            templates = set()
            for phrase in self._split_phrases(caption):
                template = templatize(phrase)
                template_to_phrases[template].add(phrase)
                templates.add(template)
            if templates:
                caption_template_sets.append(frozenset(templates))

        templates = list(template_to_phrases)

        # Pairs of templates that ever appear together -> must NOT be the same topic.
        co_occurring = set()
        for template_set in caption_template_sets:
            for a, b in combinations(sorted(template_set), 2):
                co_occurring.add((a, b))

        # Cluster templates into topics with union-find.
        union_find = _UnionFind(templates)
        members: Dict[str, set] = {t: {t} for t in templates}

        # Candidate merges: templates sharing >=1 content word, most-shared first.
        candidates = []
        for a, b in combinations(templates, 2):
            shared = _content_words(a) & _content_words(b)
            if shared:
                candidates.append((len(shared), a, b))
        candidates.sort(key=lambda c: c[0], reverse=True)

        def components_conflict(root_a, root_b) -> bool:
            return any(
                tuple(sorted((x, y))) in co_occurring
                for x in members[root_a]
                for y in members[root_b]
            )

        for _, a, b in candidates:
            root_a, root_b = union_find.find(a), union_find.find(b)
            if root_a == root_b:
                continue
            if not components_conflict(root_a, root_b):
                union_find.union(a, b)
                new_root = union_find.find(a)
                members[new_root] = members[root_a] | members[root_b]

        # Collect final topics: each is a sorted list of real phrases to sample from.
        topic_phrases: Dict[str, set] = defaultdict(set)
        for template in templates:
            root = union_find.find(template)
            topic_phrases[root].update(template_to_phrases[template])

        self.topics: List[List[str]] = [sorted(phrases) for phrases in topic_phrases.values()]

        # Empirical distribution of how many distinct topics a caption uses, so
        # generated captions resemble real ones in length.
        self._topic_counts = [
            len({union_find.find(t) for t in template_set})
            for template_set in caption_template_sets
        ] or [1]

        # Signatures of the training captions, for novelty checking.
        self.training_signatures = {
            self.signature(caption) for caption in captions
        }

    @staticmethod
    def _split_phrases(caption: str) -> List[str]:
        return [p.strip() for p in caption.split(".") if p.strip()]

    def signature(self, caption: str) -> frozenset:
        """Order- and plural-independent signature used to compare captions."""
        return frozenset(normalize_phrase(p) for p in self._split_phrases(caption))

    def generate_sentence(self, min_topics: Optional[int] = None, max_topics: Optional[int] = None) -> str:
        """Generate one random caption.

        By default the number of topics is sampled from the distribution observed
        in the training captions; ``min_topics`` / ``max_topics`` clamp that count.
        """
        num_topics = random.choice(self._topic_counts)
        if min_topics is not None:
            num_topics = max(num_topics, min_topics)
        if max_topics is not None:
            num_topics = min(num_topics, max_topics)
        num_topics = max(1, min(num_topics, len(self.topics)))

        chosen_topics = random.sample(self.topics, num_topics)
        selected_phrases = [random.choice(topic) for topic in chosen_topics]
        random.shuffle(selected_phrases)
        return ". ".join(selected_phrases) + "."

    def describe_topics(self) -> str:
        """Human-readable summary of the learned topics (for debugging)."""
        lines = [f"Learned {len(self.topics)} topics:"]
        for i, phrases in enumerate(sorted(self.topics)):
            sample = ", ".join(phrases[:5])
            more = "" if len(phrases) <= 5 else f", ... (+{len(phrases) - 5})"
            lines.append(f"  Topic {i}: {sample}{more}")
        return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/MM_LevelsAndCaptions-simple.json"
    with open(path) as f:
        data = json.load(f)
    captions = [item["caption"] for item in data]

    generator = GenericGrammarGenerator(captions, seed=512)
    print(generator.describe_topics())
    print("\nGenerated sample captions:")
    for _ in range(10):
        print(f"- {generator.generate_sentence()}")
