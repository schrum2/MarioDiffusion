import random
from typing import Dict, Optional


class GrammarGenerator:
    def __init__(self, seed=512, describe_absence=False, no_upside_down_pipes=False):
        random.seed(seed)
        self.describe_absence = describe_absence

        self.topic_phrases = {
            "floor": [
                "full floor",
                "floor with one gap",
                "floor with two gaps",
                "floor with a few gaps",
                "floor with several gaps",
                "giant gap with one chunk of floor",
                "giant gap with two chunks of floor",
                "giant gap with a few chunks of floor",
            ],
            "ceiling": [
                "full ceiling",
                "ceiling with one gap",
                "ceiling with two gaps",
                "ceiling with a few gaps",
            ],
            "wall": [
                "one wall",
                "two walls",
                "a few walls",
                "several walls",
            ],
            "enem": [
                "one enemy",
                "two enemies",
                "a few enemies",
                "several enemies",
                "many enemies",
            ],
            "powerup": [
                "one powerup",
                "two powerups",
                "a few powerups",
                "several powerups",
            ],
            "hazard": [
                "one hazard",
                "two hazards",
                "a few hazards",
                "several hazards",
            ],
            "dissapearing block": [
                "one dissapearing block",
                "two dissapearing blocks",
                "a few dissapearing blocks",
            ],
            "water": [
                "some water",
                "half water",
                "mostly water",
                "all water",
            ],
            "moving platform": [
                "one moving platform",
                "two moving platforms",
                "a few moving platforms",
                "several moving platforms",
            ],
            "platform": [
                "one platform",
                "two platforms",
                "a few platforms",
                "several platforms",
            ],
            "ladder": [
                "one ladder",
                "two ladders",
                "a few ladders",
                "several ladders",
            ],
            "tower": [
                "one tower",
                "two towers",
                "a few towers",
            ],
            "rectangular block cluster": [
                "one rectangular block cluster",
                "two rectangular block clusters",
                "a few rectangular block clusters",
            ],
            "irregular block cluster": [
                "one irregular block cluster",
                "two irregular block clusters",
                "a few irregular block clusters",
            ],
            "loose block": [
                "one loose block",
                "two loose blocks",
                "a few loose blocks",
                "several loose blocks",
            ],
        }

        self.absence_phrases = {
            "floor": "no floor",
            "ceiling": "no ceiling",
            "wall": "no walls",
            "enem": "no enemies",
            "powerup": "no powerups",
            "hazard": "no hazards",
            "dissapearing block": "no dissapearing blocks",
            "water": "no water",
            "moving platform": "no moving platforms",
            "platform": "no platforms",
            "ladder": "no ladders",
            "tower": "no towers",
            "rectangular block cluster": "no rectangular block clusters",
            "irregular block cluster": "no irregular block clusters",
            "loose block": "no loose blocks",
        }

        self.topic_keywords = [
            "floor",
            "ceiling",
            "wall",
            "enem",
            "powerup",
            "hazard",
            "dissapearing block",
            "water",
            "moving platform",
            "platform",
            "ladder",
            "tower",
            "rectangular block cluster",
            "irregular block cluster",
            "loose block",
        ]

        self.exclusive_groups = []

    def get_topic_from_phrase(self, phrase: str) -> Optional[str]:
        for keyword in self.topic_keywords:
            if keyword in phrase:
                return keyword
        return None

    def generate_sentence(self, min_topics: int = 1, max_topics: int = 10) -> str:
        num_topics = random.randint(min_topics, max_topics)
        available_topics = self.topic_keywords.copy()
        used_topics = set()
        selected_phrases = []

        for _ in range(num_topics):
            if not available_topics:
                break

            topic = random.choice(available_topics)
            available_topics.remove(topic)
            used_topics.add(topic)

            for group in self.exclusive_groups:
                if topic in group:
                    for exclusive_topic in group:
                        if exclusive_topic in available_topics and exclusive_topic != topic:
                            available_topics.remove(exclusive_topic)

            phrase = random.choice(self.topic_phrases[topic])
            selected_phrases.append(phrase)

        if self.describe_absence:
            for topic in self.topic_keywords:
                if topic not in used_topics and topic in self.absence_phrases:
                    selected_phrases.append(self.absence_phrases[topic])

        random.shuffle(selected_phrases)
        return ". ".join(selected_phrases) + "."

    def parse_sentence(self, sentence: str) -> Dict[str, str]:
        result = {}
        phrases = [p.strip() for p in sentence.strip(".").split(".")]
        for phrase in phrases:
            topic = self.get_topic_from_phrase(phrase)
            if topic:
                result[topic] = phrase
        return result

    def is_valid_sentence(self, sentence: str) -> bool:
        phrases = [p.strip() for p in sentence.strip(".").split(".")]
        seen_topics = set()
        for phrase in phrases:
            phrase_topic = self.get_topic_from_phrase(phrase)
            if not phrase_topic:
                return False
            if phrase_topic in seen_topics:
                return False
            for group in self.exclusive_groups:
                if phrase_topic in group:
                    if any(topic in seen_topics for topic in group if topic != phrase_topic):
                        return False
            seen_topics.add(phrase_topic)
        return True


if __name__ == "__main__":
    generator = GrammarGenerator(seed=512, describe_absence=False)
    print("Generated sentences:")
    for _ in range(5):
        sentence = generator.generate_sentence()
        print(f"- {sentence}")