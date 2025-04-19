import random
from typing import List, Dict, Any, Optional, Set


class GrammarGenerator:
    def __init__(self):
        # Define topics and their valid variations
        self.topic_phrases = {
            "floor": ["full floor", "floor with one gap", "floor with two gaps", 
                     "giant gap with one chunk of floor", "giant gap with two chunks of floor"],
            "ceiling": ["full ceiling", "ceiling with one gap"],
            "broken pipe": ["one broken pipe", "two broken pipes"],
            "pipe": ["one pipe", "two pipes", "a few pipes"],
            "coin line": ["one coin line", "two coin lines", "a few coin lines"],
            "coin": ["one coin", "two coins", "several coins", "a few coins"],
            "platform": ["one platform", "two platforms", "a few platforms"],
            "tower": ["one tower", "two towers"],
            "wall": ["one wall", "two walls"],
            "broken cannon": ["one broken cannon", "two broken cannons"],
            "cannon": ["one cannon", "two cannons"],
            "ascending staircase": ["one ascending staircase", "two ascending staircases"],
            "descending staircase": ["one descending staircase", "two descending staircases"],
            "irregular": ["one irregular block cluster", "two irregular block clusters"],
            "question block": ["one question block", "two question blocks", "several question blocks", "a few question blocks"],
            "enem": ["one enemy", "two enemies", "a few enemies", "several enemies"]
        }
        
        # These are the keywords used to identify topics
        self.topic_keywords = [
            "floor", "ceiling", 
            "broken pipe", "pipe", 
            "coin line", "coin",
            "platform", "tower", "wall", 
            "broken cannon", "cannon",
            "ascending staircase", "descending staircase",
            "irregular", "question block", "enem"
        ]
        
        # Define topic groups that are mutually exclusive
        self.exclusive_groups = [
            {"broken pipe", "pipe"},
            {"broken cannon", "cannon"}
        ]

    def get_topic_from_phrase(self, phrase: str) -> Optional[str]:
        """Identify which topic a phrase belongs to."""
        for keyword in self.topic_keywords:
            if keyword in phrase:
                return keyword
        return None

    def generate_sentence(self, min_topics: int = 3, max_topics: int = 6) -> str:
        """Generate a random sentence with a specified number of topics."""
        # Decide how many topics to include
        num_topics = random.randint(min_topics, max_topics)
        
        # Make a copy of available topics
        available_topics = self.topic_keywords.copy()
        
        # Track used topics to respect exclusive relationships
        used_topics = set()
        
        # Collect the phrases for our sentence
        selected_phrases = []
        
        for _ in range(num_topics):
            if not available_topics:
                break
                
            # Select a random topic
            topic = random.choice(available_topics)
            available_topics.remove(topic)
            used_topics.add(topic)
            
            # Remove any topics that are exclusive with the selected topic
            for group in self.exclusive_groups:
                if topic in group:
                    for exclusive_topic in group:
                        if exclusive_topic in available_topics and exclusive_topic != topic:
                            available_topics.remove(exclusive_topic)
            
            # Select a random phrase for this topic
            phrase = random.choice(self.topic_phrases[topic])
            selected_phrases.append(phrase)
        
        # Shuffle the phrases and join with periods
        random.shuffle(selected_phrases)
        return ". ".join(selected_phrases) + "."

    def parse_sentence(self, sentence: str) -> Dict[str, str]:
        """Parse a sentence into its component topics and phrases."""
        result = {}
        phrases = [p.strip() for p in sentence.strip(".").split(".")]
        
        for phrase in phrases:
            topic = self.get_topic_from_phrase(phrase)
            if topic:
                result[topic] = phrase
                
        return result

    def is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence follows the grammar rules."""
        phrases = [p.strip() for p in sentence.strip(".").split(".")]
        
        # Track which topics we've seen
        seen_topics = set()
        
        for phrase in phrases:
            # Find which topic this phrase belongs to
            phrase_topic = self.get_topic_from_phrase(phrase)
            
            # If no valid topic, this is invalid
            if not phrase_topic:
                return False
                
            # Check if we've already seen this topic
            if phrase_topic in seen_topics:
                return False
                
            # Check exclusive groups
            for group in self.exclusive_groups:
                if phrase_topic in group:
                    # If we've seen another topic from this exclusive group, invalid
                    if any(topic in seen_topics for topic in group if topic != phrase_topic):
                        return False
            
            seen_topics.add(phrase_topic)
            
        return True


# Example usage
if __name__ == "__main__":
    generator = GrammarGenerator()
    
    # Generate random sentences
    print("Generated sentences:")
    for _ in range(5):
        sentence = generator.generate_sentence()
        print(f"- {sentence}")
    
    # Test with example sentences
    example_sentences = [
        "full floor. one enemy. a few question blocks. one platform. one pipe.",
        "full floor. one enemy. two pipes.",
        "floor with one gap. one enemy. one question block. two platforms.",
        "full floor. a few enemies. two question blocks. two platforms.",
        "full floor. full ceiling. two enemies. several question blocks. one platform. one irregular block cluster.",
        "floor with one gap. full ceiling. two enemies. one irregular block cluster. one tower.",
        "giant gap with one chunk of floor. two platforms.",
        "giant gap with two chunks of floor. one enemy. one question block. two coins. one coin line. two platforms.",
        "giant gap with one chunk of floor. one enemy. several coins. two coin lines. a few platforms.",
        "full floor. a few enemies. one cannon.",
        "full floor. two enemies. one cannon. one ascending staircase."
    ]
    
    print("\nValidation of example sentences:")
    for sentence in example_sentences:
        is_valid = generator.is_valid_sentence(sentence)
        print(f"- {'✓' if is_valid else '✗'} {sentence}")
        if not is_valid:
            print(f"  Topics found: {generator.parse_sentence(sentence)}")
    
    # Test custom sentence
    custom_sentence = "full floor. one pipe. one broken pipe."  # This should be invalid due to exclusive groups
    print(f"\nCustom test - '{custom_sentence}': {'Valid' if generator.is_valid_sentence(custom_sentence) else 'Invalid'}")
