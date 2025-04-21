
# Quantity order for scoring partial matches
QUANTITY_TERMS = ["one", "two", "a few", "several", "many"]

# Topics to compare
TOPIC_KEYWORDS = [
    "giant gap", 
    "floor", "ceiling", 
    "broken pipe", "pipe", 
    "coin line", "coin",
    "platform", "tower", "wall", 
    "broken cannon", "cannon",
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
        if word.endswith('s') and not word.endswith('ss'):  # avoid "class", "boss"
            singular = word[:-1]
            normalized_words.append(singular)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

def extract_phrases(caption, debug=False):
    phrases = [phrase.strip() for phrase in caption.split('.') if phrase.strip()]
    topic_to_phrase = {}
    already_matched_phrases = set()  # Track phrases that have been matched
    
    for topic in TOPIC_KEYWORDS:
        matching_phrases = []
        
        for p in phrases:
            # Only consider phrases that haven't been matched to longer topics
            if topic in p and p not in already_matched_phrases:
                matching_phrases.append(p)
        
        if matching_phrases:
            # Filter out "no ..." phrases as equivalent to absence
            phrase = matching_phrases[0]
            if phrase.lower().startswith("no "):
                topic_to_phrase[topic] = None
                if debug:
                    print(f"[Extract] Topic '{topic}': detected 'no ...', treating as None")
            else:
                topic_to_phrase[topic] = phrase
                already_matched_phrases.add(phrase)  # Mark this phrase as matched
                if debug:
                    print(f"[Extract] Topic '{topic}': found phrase '{phrase}'")
        else:
            topic_to_phrase[topic] = None
            if debug:
                print(f"[Extract] Topic '{topic}': no phrase found")
    
    return topic_to_phrase

def quantity_score(phrase1, phrase2, debug=False):
    def find_quantity(phrase):
        for term in QUANTITY_TERMS:
            if term in phrase:
                return term
        return None

    qty1 = find_quantity(phrase1)
    qty2 = find_quantity(phrase2)

    if debug:
        print(f"[Quantity] Comparing quantities: '{qty1}' vs. '{qty2}'")

    if qty1 and qty2:
        idx1 = QUANTITY_TERMS.index(qty1)
        idx2 = QUANTITY_TERMS.index(qty2)
        diff = abs(idx1 - idx2)
        max_diff = len(QUANTITY_TERMS) - 1
        score = 1.0 - (diff / max_diff)
        if debug:
            print(f"[Quantity] Quantity indices: {idx1} vs. {idx2}, diff: {diff}, score: {score:.2f}")
        return score
    if debug:
        print("[Quantity] At least one quantity missing, assigning partial score 0.1")
    return 0.1

def compare_captions(correct_caption, generated_caption, debug=False):
    correct_phrases = extract_phrases(correct_caption, debug=debug)
    generated_phrases = extract_phrases(generated_caption, debug=debug)

    total_score = 0.0
    num_topics = len(TOPIC_KEYWORDS)

    if debug:
        print("\n--- Starting Topic Comparison ---\n")

    for topic in TOPIC_KEYWORDS:
        correct = correct_phrases[topic]
        generated = generated_phrases[topic]

        if debug:
            print(f"[Topic: {topic}] Correct: {correct} | Generated: {generated}")

        if correct is None and generated is None:
            total_score += 1.0
            if debug:
                print(f"[Topic: {topic}] Both None — full score: 1.0\n")
        elif correct is None or generated is None:
            total_score += -1.0
            if debug:
                print(f"[Topic: {topic}] One is None — penalty: -1.0\n")
        else:
            # Normalize pluralization before comparison
            norm_correct = normalize_plural(correct)
            norm_generated = normalize_plural(generated)

            if debug:
                print(f"[Topic: {topic}] Normalized: Correct: '{norm_correct}' | Generated: '{norm_generated}'")

            if norm_correct == norm_generated:
                total_score += 1.0
                if debug:
                    print(f"[Topic: {topic}] Exact match — score: 1.0\n")
            elif any(term in norm_correct for term in QUANTITY_TERMS) and any(term in norm_generated for term in QUANTITY_TERMS):
                qty_score = quantity_score(norm_correct, norm_generated, debug=debug)
                total_score += qty_score
                if debug:
                    print(f"[Topic: {topic}] Quantity-based partial score: {qty_score:.2f}\n")
            else:
                total_score += 0.1
                if debug:
                    print(f"[Topic: {topic}] Partial match (topic overlap) — score: 0.1\n")

    final_score = total_score / num_topics
    if debug:
        print(f"--- Final score: {final_score:.4f} ---\n")
    return final_score

if __name__ == '__main__':

    ref = "floor with one gap. two enemies. one platform. one tower."
    gen = "giant gap with one chunk of floor. two enemies. one platform. one tower."

    score = compare_captions(ref, gen, debug=True)
    print(f"Should be: {ref}")
    print(f"  but was: {gen}")
    print(f"Score: {score}")
