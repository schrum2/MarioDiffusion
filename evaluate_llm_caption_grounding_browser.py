"""Browse and inspect output from evaluate_llm_caption_grounding.py.

Example:
    python evaluate_llm_caption_grounding_browser.py \
        --input scored.json --game MMLV

The scene grid, tile decoding, navigation, filtering, and window behavior are
inherited from ascii_data_browser.TileViewer.  This viewer adds a compact review
panel showing the selected caption and the concepts that supported or hurt its score.
"""

import argparse
import json

import tkinter as tk

from ascii_data_browser import TileViewer
from util.common_settings import GAME_CLI_CHOICES
from captions.util import extract_tileset


class GroundingReviewViewer(TileViewer):
    """TileViewer with a score explanation panel for grounding-evaluation JSON."""

    def __init__(self, dataset_path, game):
        super().__init__(dataset_path=dataset_path, game=game)

        self.attr_var.set("scores")
        self.current_caption_idx = 0

        review_frame = tk.LabelFrame(self.scroll_frame, text="Grounding Review")
        review_frame.pack(fill=tk.X, padx=4, pady=(2, 8))
        self.review_text = tk.Text(review_frame, height=10, width=100, wrap=tk.WORD,
                                   state=tk.DISABLED)
        self.review_text.pack(fill=tk.X, padx=4, pady=4)
        self.redraw()

    def load_files_from_paths(self, dataset_path, tileset_path):
        """Load the evaluator's summary object, then initialize TileViewer state."""
        with open(dataset_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict) or not isinstance(summary.get("entries"), list):
            raise ValueError("Input must contain an 'entries' list from the grounding evaluator")

        self.dataset_path = dataset_path
        self.dataset = summary["entries"]
        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
        self.color_map = self._build_color_map()
        self.current_sample_idx = 0
        self.current_caption_idx = 0
        self.filter_text = ""
        self.filtered_indexes = None
        if hasattr(self, "filter_var"):
            self.filter_var.set("")

    def _selected_values(self, sample):
        """Show caption strings in the inherited caption box for the scores field."""
        if self.attr_var.get() == "scores" and isinstance(sample, dict):
            return [score.get("caption", "") for score in sample.get("scores", [])]
        return super()._selected_values(sample)

    def _review_value(self, sample):
        scores = sample.get("scores", []) if isinstance(sample, dict) else []
        if not scores:
            return None
        index = min(self.current_caption_idx, len(scores) - 1)
        return scores[index]

    def _format_review(self, score):
        if score is None:
            return "No score is available for this scene."

        def names(items):
            if not items:
                return "(none)"
            return ", ".join(
                item if isinstance(item, str) else f"{item.get('char', '?')}: {item.get('description', '')}"
                for item in items
            )

        return (
            f"Overall: {score.get('overall', 'n/a')}    "
            f"Coverage: {score.get('coverage', 'n/a')}    "
            f"Precision: {score.get('precision', 'n/a')}\n\n"
            f"Caption:\n{score.get('caption', '')}\n\n"
            f"Scene categories present:\n{names(score.get('present_categories', []))}\n\n"
            f"Categories mentioned:\n{names(score.get('mentioned_categories', []))}\n\n"
            f"Unsupported categories:\n{names(score.get('unsupported_categories', []))}\n\n"
            f"Supported specific tiles:\n{names(score.get('supported_specific_tiles', []))}\n\n"
            f"Unsupported specific tiles:\n{names(score.get('unsupported_specific_tiles', []))}\n\n"
            f"Scene tile counts:\n{names([f'{char}: {count}' for char, count in score.get('scene_tile_counts', {}).items()])}"
        )

    def redraw(self):
        super().redraw()
        if not hasattr(self, "review_text") or not self.dataset:
            return
        score = self._review_value(self.dataset[self.current_sample_idx])
        self.review_text.configure(state=tk.NORMAL)
        self.review_text.delete("1.0", tk.END)
        self.review_text.insert("1.0", self._format_review(score))
        self.review_text.configure(state=tk.DISABLED)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Scored JSON from evaluate_llm_caption_grounding.py")
    parser.add_argument("--game", required=True, choices=GAME_CLI_CHOICES,
                        help="Game used when producing the scored JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        scored = json.load(handle)
    if not isinstance(scored, dict) or not isinstance(scored.get("entries"), list):
        raise ValueError("Input must be the summary JSON produced by evaluate_llm_caption_grounding.py")

    viewer = GroundingReviewViewer(args.input, args.game)
    viewer.mainloop()


if __name__ == "__main__":
    main()