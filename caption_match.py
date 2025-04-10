def compare_captions(reference_caption, generated_caption):
    topics = [
        "floor", "ceiling", "pipe", "coin line", "coin", "platform",
        "tower", "wall", "cannon", "ascending staircase",
        "descending staircase", "irregular", "question block", "enem"
    ]
    quantity_terms = ["one", "two", "a few", "several", "many"]

    def phrase_map(caption):
        """Map topics to phrases from the caption."""
        phrases = [p.strip() for p in caption.split(".") if p.strip()]
        topic_to_phrase = {}
        for topic in topics:
            for phrase in phrases:
                # Exact match for 'coin line' before 'coin'
                if topic == "coin line" and "coin line" in phrase:
                    topic_to_phrase[topic] = phrase
                    break
                elif topic != "coin line" and topic in phrase:
                    topic_to_phrase[topic] = phrase
                    break
        return topic_to_phrase

    def is_negative(phrase):
        return phrase.lower().startswith("no ")

    ref_map = phrase_map(reference_caption)
    gen_map = phrase_map(generated_caption)

    total_score = 0.0
    num_topics = len(topics)

    for topic in topics:
        ref_phrase = ref_map.get(topic)
        gen_phrase = gen_map.get(topic)

        # Normalize "no" phrases to None for consistency
        if ref_phrase and is_negative(ref_phrase):
            ref_phrase = None
        if gen_phrase and is_negative(gen_phrase):
            gen_phrase = None

        if ref_phrase is None and gen_phrase is None:
            # Neither mentions the topic: neutral score
            score = 0.0
        elif ref_phrase is not None and gen_phrase is not None:
            if ref_phrase == gen_phrase:
                score = 1.0
            else:
                # Check for quantity difference
                ref_quantity = next((q for q in quantity_terms if q in ref_phrase), None)
                gen_quantity = next((q for q in quantity_terms if q in gen_phrase), None)

                if (ref_quantity and gen_quantity and
                    ref_phrase.replace(ref_quantity, "").strip() == gen_phrase.replace(gen_quantity, "").strip()):
                    # Only quantity difference
                    distance = abs(quantity_terms.index(ref_quantity) - quantity_terms.index(gen_quantity))
                    max_distance = len(quantity_terms) - 1
                    score = 1.0 - (distance / max_distance) * 0.75
                    score = max(score, 0.25)  # Ensure lower bound of 0.25
                else:
                    # Other differences
                    score = 0.1
        elif ref_phrase is not None and gen_phrase is None:
            # Reference mentions topic, generated does not
            score = -1.0
        else:  # ref_phrase is None and gen_phrase is not None
            # Generated mentions topic, reference does not
            score = -1.0

        total_score += score

    return total_score / num_topics
