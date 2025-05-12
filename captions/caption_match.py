from create_ascii_captions import assign_caption

# Quantity order for scoring partial matches
QUANTITY_TERMS = ["one", "two", "a few", "several", "many"]

# Topics to compare
TOPIC_KEYWORDS = [
    #"giant gap", # I think all gaps are subsumed by the floor topic 
    "floor", "ceiling", 
    "broken pipe", "upside down pipe", "pipe", 
    "coin line", "coin",
    "platform", "tower", #"wall", 
    "broken cannon", "cannon",
    "ascending staircase", "descending staircase",
    "rectangular",
    "irregular", 
    "question block", "loose block",
    "enem"  # catch "enemy"/"enemies"
]

BROKEN_TOPICS = 2 # Number of topics that are considered "broken" (e.g., "broken pipe", "broken cannon")

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

def compare_captions(correct_caption, generated_caption, debug=False, return_matches=False):
    correct_phrases = extract_phrases(correct_caption, debug=debug)
    generated_phrases = extract_phrases(generated_caption, debug=debug)

    total_score = 0.0
    num_topics = len(TOPIC_KEYWORDS)

    exact_matches = []
    partial_matches = []
    excess_phrases = []

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
            if generated is not None:
                excess_phrases.append(generated)
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
                exact_matches.append(generated)
                if debug:
                    print(f"[Topic: {topic}] Exact match — score: 1.0\n")
            elif any(term in norm_correct for term in QUANTITY_TERMS) and any(term in norm_generated for term in QUANTITY_TERMS):
                qty_score = quantity_score(norm_correct, norm_generated, debug=debug)
                total_score += qty_score
                partial_matches.append(generated)
                if debug:
                    print(f"[Topic: {topic}] Quantity-based partial score: {qty_score:.2f}\n")
            else:
                total_score += 0.1
                partial_matches.append(generated)
                if debug:
                    print(f"[Topic: {topic}] Partial match (topic overlap) — score: 0.1\n")

    final_score = total_score / num_topics
    if debug:
        print(f"--- Final score: {final_score:.4f} ---\n")

    if return_matches:
        return final_score, exact_matches, partial_matches, excess_phrases

    return final_score

def process_scene_segments(scene, segment_width, prompt, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, verbose=False):
    """
    Process a scene by partitioning it into segments, assigning captions, and computing comparison scores.

    Args:
        scene (list): The scene to process, represented as a 2D list.
        segment_width (int): The width of each segment.
        prompt (str): The prompt to compare captions against.
        id_to_char (dict): Mapping from tile IDs to characters.
        char_to_id (dict): Mapping from characters to tile IDs.
        tile_descriptors (dict): Descriptions of individual tile types.
        describe_locations (bool): Whether to include location descriptions in captions.
        describe_absence (bool): Whether to indicate absence of items in captions.
        verbose (bool): If True, print captions and scores for each segment.

    Returns:
        tuple: A tuple containing the average comparison score, captions for each segment, and scores for each segment.
    """
    # Partition the scene into segments of the specified width
    segments = [
        [row[i:i+segment_width] for row in scene]  # Properly slice each row of the scene
        for i in range(0, len(scene[0]), segment_width)
    ]

    # Assign captions and compute scores for each segment
    segment_scores = []
    segment_captions = []
    for idx, segment in enumerate(segments):
        segment_caption = assign_caption(segment, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence)
        segment_score = compare_captions(prompt, segment_caption)
        segment_scores.append(segment_score)
        segment_captions.append(segment_caption)

        if verbose:
            print(f"Segment {idx + 1} caption: {segment_caption}")
            print(f"Segment {idx + 1} comparison score: {segment_score}")

    # Compute the average comparison score
    average_score = sum(segment_scores) / len(segment_scores) if segment_scores else 0

    if verbose:
        print(f"Average comparison score across all segments: {average_score}")

    return average_score, segment_captions, segment_scores

if __name__ == '__main__':

    ref = "floor with one gap. two enemies. one platform. one tower."
    gen = "giant gap with one chunk of floor. two enemies. one platform. one tower."

    score = compare_captions(ref, gen, debug=True)
    print(f"Should be: {ref}")
    print(f"  but was: {gen}")
    print(f"Score: {score}")
