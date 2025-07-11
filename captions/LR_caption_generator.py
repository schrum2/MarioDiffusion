import random
from typing import Dict, Optional


class GrammarGenerator:
    def __init__(self, seed=512, describe_absence=False, no_upside_down_pipes=False):
        random.seed(seed)  # Set the random seed for reproducibility
        self.describe_absence = describe_absence
        self.no_upside_down_pipes = no_upside_down_pipes

        # Define topics and their valid variations
        self.topic_phrases = {
            "floor": ["full floor", "floor with one gap", "floor with two gaps", "floor with a few gaps", "floor with several gaps",
                     "giant gap with one chunk of floor", "giant gap with two chunks of floor", "giant gap with a few chunks of floor", "giant gap with several chunks of floor"],
            # "ceiling": ["full ceiling", "ceiling with one gap", "ceiling with two gaps", "ceiling with a few gaps"],
            # "broken pipe": ["one broken pipe", "two broken pipes"],
            "gold line": ["one gold line", "two gold lines", "a few gold lines"],
            "gold": ["one gold", "two gold", "several gold", "a few gold", "many gold"],
            #"platform": ["one platform", "two platforms", "a few platforms", "several platforms"],
            "ladder cluster": ["one ladder cluster", "two ladder clusters", "a few ladder clusters", "several ladder clusters"],
            "lone ladder tile": ["one lone ladder tile", "two lone ladder tiles", "a few lone ladder tiles", "several lone ladder tiles"],
            "short ladder": ["one short ladder", "two short ladders", "a few short ladders", "several short ladders"],
            "tall ladder": ["one tall ladder", "two tall ladders", "a few tall ladders", "several tall ladders"],
            "chamber": ["one chamber", "two chambers", "several chambers", "a few chambers", "many chambers"],
            "rope": ["one rope", "two ropes", "several ropes", "a few ropes", "many ropes"],
            "rectangular": ["one rectangular block cluster", "two rectangular block clusters", "a few rectangular block clusters"],
            "irregular": ["one irregular block cluster", "two irregular block clusters", "a few irregular block clusters"],
            "loose block": ["one loose block", "two loose blocks", "several loose blocks", "a few loose blocks", "many loose blocks"],
            "diggable ground": ["one diggable ground", "two diggable ground", "several diggable ground", "a few diggable ground", "many diggable ground"],
            "solid ground": ["one solid ground", "two solid ground", "several solid ground", "a few solid ground", "many solid ground"],
            "background area": ["one background area", "two background areas", "several background areas", "a few background areas", "many background areas"],
            "enem": ["one enemy", "two enemies", "a few enemies", "several enemies"]
        }

        # Topic absence descriptions
        self.absence_phrases = {
            "floor": "no floor",
            # "ceiling": "no ceiling",
            "gold line": "no gold lines", 
            "gold": "no gold",
            "ladder cluster": "no ladder clusters",
            "lone ladder tile": "no lone ladder tiles",
            "short ladder": "no short ladders",
            "tall ladder": "no tall ladders",
            "chamber": "no chamber",
            "rope": "no ropes",
            #"platform": "no platforms",
            "rectangular": "no rectangular block clusters",
            "irregular": "no irregular block clusters",
            "loose block": "no loose blocks",
            "diggable ground": "no diggable ground",
            "solid ground": "no solid ground",
            "background area": "no background area",
            "enem": "no enemies"
        }
        
        # These are the keywords used to identify topics
        self.topic_keywords = [
            "floor", # "ceiling",
            "gold line", "gold",
            "ladder cluster",
            "lone ladder tile", "short ladder", "tall ladder",
            "chamber",
            "rope",
            #"platform",
            "rectangular",
            "irregular", 
            "loose block",
            "diggable ground",
            "solid ground",
            "background area", 
            "enem"
        ]

        # Remove upside down pipes if specified
        if self.no_upside_down_pipes:
            self.topic_phrases.pop("upside down pipe", None)
            self.absence_phrases.pop("upside down pipe", None)
            self.topic_keywords.remove("upside down pipe")
        
        # Define topic groups that are mutually exclusive
        # Wrong: there can be a mix of valid and broken pipes
        #self.exclusive_groups = [
        #    {"broken pipe", "pipe"},
        #    {"broken cannon", "cannon"}
        #]
        self.exclusive_groups = []

    def get_topic_from_phrase(self, phrase: str) -> Optional[str]:
        """Identify which topic a phrase belongs to."""
        for keyword in self.topic_keywords:
            if keyword in phrase:
                return keyword
        return None

    def generate_sentence(self, min_topics: int = 1, max_topics: int = 10) -> str:
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
            
            # Remove any topics that are exclusive with the selected topic (should not be needed, but doesn't hurt)
            for group in self.exclusive_groups:
                if topic in group:
                    for exclusive_topic in group:
                        if exclusive_topic in available_topics and exclusive_topic != topic:
                            available_topics.remove(exclusive_topic)
            
            # Select a random phrase for this topic
            phrase = random.choice(self.topic_phrases[topic])
            selected_phrases.append(phrase)

        # Special case for consistenct of gold and gold lines
        if "gold line" in used_topics and "gold" not in used_topics:
            # If coin line is present, add a coin
            selected_phrases.append(random.choice(self.topic_phrases["gold"]))
            used_topics.add("gold")

        # If describe_absence is True, add absence descriptions for unused topics
        if self.describe_absence:
            for topic in self.topic_keywords:
                if topic not in used_topics and topic in self.absence_phrases:
                    selected_phrases.append(self.absence_phrases[topic])
        
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
    # Test regular generation
    generator = GrammarGenerator(seed=512, describe_absence=False)
    print("Generated sentences without absence descriptions:")
    for _ in range(3):
        sentence = generator.generate_sentence(min_topics=2, max_topics=4)
        print(f"- {sentence}")
    
    # Test generation with absence descriptions
    generator_with_absence = GrammarGenerator(seed=512, describe_absence=True)
    print("\nGenerated sentences with absence descriptions:")
    for _ in range(3):
        sentence = generator_with_absence.generate_sentence(min_topics=2, max_topics=4)
        print(f"- {sentence}")
    
    # Rest of the test code...
    generator = GrammarGenerator()
    
    # Generate random sentences
    print("Generated sentences:")
    for _ in range(5):
        sentence = generator.generate_sentence()
        print(f"- {sentence}")
    
    # Test with example sentences
    lr_example_sentences = [
        "full floor. one enemy. a few ladders. one rope.",
        "full floor. one enemy. two ropes.",
        "floor with one gap. one enemy. one gold.",
        "full floor. a few enemies. two ladders.",
        "full floor. two enemies. several ladders. one irregular block cluster.",
        "floor with one gap. two enemies. one irregular block cluster. one gold.",
        "giant gap with one chunk of floor.",
        "giant gap with two chunks of floor. one enemy. one ladder. two gold. one gold line.",
        "giant gap with one chunk of floor. one enemy. several gold. two gold lines.",
        "full floor. a few enemies. one ladder.",
        "full floor. two enemies. one ladder. one gold line."
    ]
    
    print("\nValidation of example sentences:")
    for sentence in lr_example_sentences:
        is_valid = generator.is_valid_sentence(sentence)
        print(f"- {'GOOD' if is_valid else ' BAD'} {sentence}")
        if not is_valid:
            print(f"  Topics found: {generator.parse_sentence(sentence)}")
    
    # Test custom sentence
    custom_sentence = "full floor. one pipe. one broken pipe."  # This should be invalid due to "broken"
    print(f"\nCustom test - '{custom_sentence}': {'Valid' if generator.is_valid_sentence(custom_sentence) else 'Invalid'}")
