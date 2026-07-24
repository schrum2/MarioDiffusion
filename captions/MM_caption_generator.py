import random
from typing import Dict, Optional


class GrammarGenerator:
    def __init__(self, seed=512, describe_absence=False, no_upside_down_pipes=False,
                 include_entrance_exit=False):
        random.seed(seed)
        self.describe_absence = describe_absence
        # Entrance/exit direction phrases are off by default since they don't reflect
        # what's actually in most captioned scenes; pass include_entrance_exit=True
        # to turn them back on.
        self.include_entrance_exit = include_entrance_exit

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
            "left wall": [
                "left wall",
                "perforated left wall",
            ],
            "right wall": [
                "right wall",
                "perforated right wall",
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
            "disappearing block": [
                "one disappearing block",
                "two disappearing blocks",
                "a few disappearing blocks",
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
                # Single-category phrases, covering all four ladder types
                "one ladder at top",
                "two ladders at top",
                "a few ladders at top",
                "one ladder at bottom",
                "two ladders at bottom",
                "a few ladders at bottom",
                "one ladder in the middle",
                "two ladders in the middle",
                "a few ladders in the middle",
                "one full height ladder",
                "two full height ladders",
                "a few full height ladders",
                # Occasional combos of two ladder types in the same scene, since real
                # scenes sometimes have more than one kind of ladder at once. Kept rare
                # relative to the single-category phrases above.
                "one ladder at top. one ladder at bottom",
                "one ladder at top. one full height ladder",
                "one ladder at bottom. one full height ladder",
                "a few ladders at bottom. one ladder at top",
                "one ladder in the middle. one full height ladder",
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
            "entrance direction": [
                "entrance direction left",
                "entrance direction bottom",
                "entrance direction top",
            ],
            "exit direction": [
                "exit direction right",
                "exit direction bottom",
                "exit direction top",
            ],
        }

        self.absence_phrases = {
            "floor": "no floor",
            "ceiling": "no ceiling",
            "left wall": "no left wall",
            "right wall": "no right wall",
            "enem": "no enemies",
            "powerup": "no powerups",
            "hazard": "no hazards",
            "disappearing block": "no disappearing blocks",
            "water": "no water",
            "moving platform": "no moving platforms",
            "platform": "no platforms",
            "ladder": "no ladders",
            "tower": "no towers",
            "rectangular block cluster": "no rectangular block clusters",
            "irregular block cluster": "no irregular block clusters",
            "loose block": "no loose blocks",
            "entrance direction": "no entrance direction",
            "exit direction": "no exit direction",
        }

        self.topic_keywords = [
            "floor",
            "ceiling",
            "left wall",
            "right wall",
            "enem",
            "powerup",
            "hazard",
            "disappearing block",
            "water",
            "moving platform",
            "platform",
            "ladder",
            "tower",
            "rectangular block cluster",
            "irregular block cluster",
            "loose block",
            "entrance direction",
            "exit direction",
        ]

        self.exclusive_groups = []
        self.horizontal_entrance = "entrance direction left"
        self.horizontal_exit = "exit direction right"
        self.vertical_directions = ["top", "bottom"]

    def get_topic_from_phrase(self, phrase: str) -> Optional[str]:
        for keyword in self.topic_keywords:
            if keyword in phrase:
                return keyword
        return None

    def generate_sentence(self, min_topics: int = 1, max_topics: int = 10) -> str:
            num_topics = random.randint(min_topics, max_topics)
            selected_phrases = []
            used_topics = set()

            if self.include_entrance_exit:
                entrance_phrase, exit_phrase = self.generate_entrance_exit()
                selected_phrases.extend([entrance_phrase, exit_phrase])
                used_topics.update({"entrance direction", "exit direction"})
            else:
                # Keep these out of the random pool entirely when the flag is off,
                # not just unselected by default.
                used_topics.update({"entrance direction", "exit direction"})

            available_topics = [t for t in self.topic_keywords if t not in used_topics]

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
                    if topic == "entrance direction" or topic == "exit direction":
                        if not self.include_entrance_exit:
                            continue
                    if topic not in used_topics and topic in self.absence_phrases:
                        selected_phrases.append(self.absence_phrases[topic])

            random.shuffle(selected_phrases)
            return ". ".join(selected_phrases) + "."

    def generate_entrance_exit(self):
        """Generate sensible entrance/exit direction pair matching training data conventions."""
        # Randomly decide if movement is horizontal or vertical
        if random.random() < 0.5:
            # Horizontal: training data always has entrance left, exit right
            return "entrance direction left", "exit direction right"
        else:
            # Vertical: entrance and exit can be top/bottom but not the same
            entrance = random.choice(["top", "bottom"])
            exit_choices = [d for d in ["top", "bottom"] if d != entrance]
            exit_dir = random.choice(exit_choices)
            return f"entrance direction {entrance}", f"exit direction {exit_dir}"

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
            # Ladder is allowed to appear as more than one phrase per sentence
            # (e.g. "one ladder at top" + "one full height ladder" as separate
            # sub-phrases from a combo entry), so don't reject repeats of it.
            if phrase_topic in seen_topics and phrase_topic != "ladder":
                return False
            for group in self.exclusive_groups:
                if phrase_topic in group:
                    if any(topic in seen_topics for topic in group if topic != phrase_topic):
                        return False
            seen_topics.add(phrase_topic)
        return True


if __name__ == "__main__":
    generator = GrammarGenerator(seed=512, describe_absence=False)
    print("Generated sentences (entrance/exit off by default):")
    for _ in range(5):
        sentence = generator.generate_sentence()
        print(f"- {sentence}")

    print("\nGenerated sentences (entrance/exit enabled):")
    generator2 = GrammarGenerator(seed=512, describe_absence=False, include_entrance_exit=True)
    for _ in range(5):
        sentence = generator2.generate_sentence()
        print(f"- {sentence}")

