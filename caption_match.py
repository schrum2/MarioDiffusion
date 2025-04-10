import re

# Quantity order for scoring partial matches
QUANTITY_TERMS = ["one", "two", "a few", "several", "many"]

# Topics to compare
TOPIC_KEYWORDS = [
    "floor", "ceiling", "pipe", "coin line", "coin",
    "platform", "tower", "wall", "cannon",
    "ascending staircase", "descending staircase",
    "irregular", "question block", "enem"  # catch "enemy"/"enemies"
]

# Plural normalization map (irregulars)
PLURAL_EXCEPTIONS = {
    "enemies": "enemy",
}

def normalize_plural(phrase):
    # Normalize known irregular plurals
    for plural, singular in PLURAL_EXCEPTIONS.items():
        phrase = phrase.replace(plural, singular)

    # Normalize regular plurals (basic "s" endings)
    words = phrase.split()
    normalized_words = []
    for word in words:
        if word.endswith('s') and not word.endswith('ss'):  # avoid "class" etc.
            singular = word[:-1]
            normalized_words.append(singular)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

def extract_phrases(caption):
    phrases = [phrase.strip() for phrase in caption.split('.') if phrase.strip()]
    topic_to_phrase = {}
    for topic in TOPIC_KEYWORDS:
        matching_phrases = [p for p in phrases if topic in p]
        if matching_phrases:
            # Filter out "no ..." phrases as equivalent to absence
            phrase = matching_phrases[0]
            if phrase.lower().startswith("no "):
                topic_to_phrase[topic] = None
            else:
                topic_to_phrase[topic] = phrase
        else:
            topic_to_phrase[topic] = None
    return topic_to_phrase

def quantity_score(phrase1, phrase2):
    # Extract quantity terms
    def find_quantity(phrase):
        for term in QUANTITY_TERMS:
            if term in phrase:
                return term
        return None

    qty1 = find_quantity(phrase1)
    qty2 = find_quantity(phrase2)

    # If both have quantity terms, compare positions
    if qty1 and qty2:
        idx1 = QUANTITY_TERMS.index(qty1)
        idx2 = QUANTITY_TERMS.index(qty2)
        diff = abs(idx1 - idx2)
        max_diff = len(QUANTITY_TERMS) - 1
        return 1.0 - (diff / max_diff)  # scaled: adjacent = high score
    return 0.1  # fallback for other differences

def compare_captions(correct_caption, generated_caption):
    correct_phrases = extract_phrases(correct_caption)
    generated_phrases = extract_phrases(generated_caption)

    total_score = 0.0
    num_topics = len(TOPIC_KEYWORDS)

    for topic in TOPIC_KEYWORDS:
        correct = correct_phrases[topic]
        generated = generated_phrases[topic]

        if correct is None and generated is None:
            total_score += 1.0
        elif correct is None or generated is None:
            total_score += -1.0
        else:
            # Normalize pluralization before comparison
            norm_correct = normalize_plural(correct)
            norm_generated = normalize_plural(generated)

            if norm_correct == norm_generated:
                total_score += 1.0
            elif any(term in norm_correct for term in QUANTITY_TERMS) and any(term in norm_generated for term in QUANTITY_TERMS):
                total_score += quantity_score(norm_correct, norm_generated)
            else:
                total_score += 0.1  # generic partial credit for same topic

    return total_score / num_topics

if __name__ == '__main__':

    ref = "floor with one gap. two enemies. one platform. tall tower."
    gen = "floor with two gap. two enemies. one platform. tall tower."

    score = compare_captions(ref, gen)
    print(f"Should be: {ref}")
    print(f"  but was: {gen}")
    print(f"Score: {score}")
