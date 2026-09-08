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

    @staticmethod
    def _breakdown(score):
        """Read new breakdown fields, or reconstruct them for older score files."""
        breakdown = score.get("score_breakdown")
        if breakdown:
            return breakdown
        mentioned_categories = score.get("mentioned_categories", [])
        unsupported_categories = score.get("unsupported_categories", [])
        supported_specific = score.get("supported_specific_tiles", [])
        unsupported_specific = score.get("unsupported_specific_tiles", [])
        mentioned_count = len(mentioned_categories) + len(supported_specific) + len(unsupported_specific)
        unsupported_count = len(unsupported_categories) + len(unsupported_specific)
        return {
            "coverage_supported_categories": len(mentioned_categories) - len(unsupported_categories),
            "coverage_present_categories": len(score.get("present_categories", [])),
            "precision_supported_mentions": mentioned_count - unsupported_count,
            "precision_total_mentions": mentioned_count,
            "precision_unsupported_mentions": unsupported_count,
        }

    def _format_review(self, score):
        if score is None:
            return "No score is available for this scene."

        breakdown = self._breakdown(score)

        def names(items):
            if not items:
                return "(none)"
            return ", ".join(
                item if isinstance(item, str) else (
                    f"{'/'.join(item.get('chars', [])) or item.get('char', '?')}: "
                    f"{'; '.join(item.get('descriptions', [])) or item.get('description', '')}"
                )
                for item in items
            )

        category_matches = score.get("category_matches", [])
        if category_matches:
            category_lines = "\n".join(
                f"  {'SUPPORTED' if item.get('supported') else 'UNSUPPORTED'} category "
                f"'{item.get('category')}' matched by: {', '.join(item.get('matched_terms', []))}"
                for item in category_matches
            )
        else:
            category_lines = "  (none)"

        specific_matches = score.get("supported_specific_tiles", []) + score.get("unsupported_specific_tiles", [])
        if specific_matches:
            specific_lines = "\n".join(
                f"  {'SUPPORTED' if item.get('supported') else 'UNSUPPORTED'} concept "
                f"matched by: "
                f"{', '.join(item.get('matched_terms', []))}\n"
                f"    tile alternatives: {', '.join(item.get('chars', []))}\n"
                f"    { '; '.join(item.get('descriptions', [])) }"
                for item in specific_matches
            )
        else:
            specific_lines = "  (none)"

        coverage_supported = breakdown.get("coverage_supported_categories", "n/a")
        coverage_present = breakdown.get("coverage_present_categories", "n/a")
        coverage_supported_concepts = breakdown.get("coverage_supported_concepts", coverage_supported)
        coverage_present_concepts = breakdown.get("coverage_present_concepts", coverage_present)
        precision_supported = breakdown.get("precision_supported_mentions", "n/a")
        precision_total = breakdown.get("precision_total_mentions", "n/a")
        precision_unsupported = breakdown.get("precision_unsupported_mentions", "n/a")
        modifier_penalty = breakdown.get("precision_modifier_penalty", "n/a")
        base_overall = breakdown.get("base_overall", score.get("base_overall", "n/a"))
        specificity_bonus = breakdown.get("specificity_bonus", score.get("specificity_bonus", "n/a"))
        coverage = score.get("coverage", "n/a")
        precision = score.get("precision", "n/a")
        overall = score.get("overall", "n/a")

        def harmonic(left, right):
            return 2 * left * right / (left + right) if left + right else 0.0

        score_changes = []
        missing_categories = score.get("missing_categories", [])
        if (missing_categories and isinstance(coverage, (int, float))
                and isinstance(precision, (int, float)) and isinstance(coverage_present_concepts, (int, float))):
            present_count = max(1, coverage_present_concepts)
            coverage_without_omission = min(1.0, coverage + 1.0 / present_count)
            increase = harmonic(coverage_without_omission, precision) - harmonic(coverage, precision)
            for category in missing_categories:
                score_changes.append(
                    f"Score decreased by {increase:.6f} because caption does not mention "
                    f"{category} in the scene."
                )
        for compound in score.get("missing_compound_concepts", []):
            if (isinstance(coverage, (int, float)) and isinstance(precision, (int, float))
                    and isinstance(coverage_present_concepts, (int, float))):
                present_count = max(1, coverage_present_concepts)
                coverage_without_omission = min(1.0, coverage + 1.0 / present_count)
                increase = harmonic(coverage_without_omission, precision) - harmonic(coverage, precision)
                score_changes.append(
                    f"Score decreased by {increase:.6f} because caption does not mention "
                    f"{compound} in the scene."
                )
        if (score.get("unsupported_categories") and isinstance(coverage, (int, float))
            and isinstance(precision, (int, float)) and isinstance(precision_total, (int, float))):
            total = max(1, precision_total)
            supported = precision_supported
            precision_without_claim = min(1.0, (supported + 1.0) / total)
            increase = harmonic(coverage, precision_without_claim) - harmonic(coverage, precision)
            for category in score["unsupported_categories"]:
                score_changes.append(
                    f"Score decreased by {increase:.6f} because caption mentions {category} "
                    "but that category is not present in the scene."
                )
        for item in score.get("unsupported_specific_tiles", []):
            if item.get("specificity_kind") == "modifier":
                modifier_penalty_value = breakdown.get("unsupported_specificity_penalty", 0.25)
                total = max(1, precision_total) if isinstance(precision_total, (int, float)) else 1
                precision_without_claim = min(1.0, (precision_supported + modifier_penalty_value) / total)
                increase = harmonic(coverage, precision_without_claim) - harmonic(coverage, precision)
                reason = "the modifier is not supported by any present tile"
            else:
                total = max(1, precision_total) if isinstance(precision_total, (int, float)) else 1
                precision_without_claim = min(1.0, (precision_supported + 1.0) / total)
                increase = harmonic(coverage, precision_without_claim) - harmonic(coverage, precision)
                reason = "the referenced tile concept is not present in the scene"
            score_changes.append(
                f"Score decreased by {increase:.6f} because caption mentions "
                f"{', '.join(item.get('matched_terms', []))}, but {reason}."
            )
        score_change_text = "\n".join(score_changes) if score_changes else "No score decreases were identified."

        return (
            f"Overall: {overall}    Coverage: {coverage}    Precision: {precision}\n"
            f"Score diagnostics:\n{score_change_text}\n\n"
            f"Coverage = {coverage_supported} supported present categories / {coverage_present} present categories = {coverage}\n"
            f"Coverage concepts including compounds = {coverage_supported_concepts} supported / "
            f"{coverage_present_concepts} present = {coverage}\n"
            f"Precision = {precision_supported} supported mentions / {precision_total} recognized mentions = {precision}\n"
            f"  ({precision_unsupported} weighted unsupported penalty; modifier penalty portion: {modifier_penalty})\n"
            f"Base overall = 2 * {coverage} * {precision} / ({coverage} + {precision}) = {base_overall}\n"
            f"Specificity bonus = {specificity_bonus}\n"
            f"Overall = base overall + specificity bonus = {overall}\n\n"
            f"Caption:\n{score.get('caption', '')}\n\n"
            f"Scene categories present:\n{names(score.get('present_categories', []))}\n\n"
            f"Categories mentioned:\n{names(score.get('mentioned_categories', []))}\n\n"
            f"Unsupported categories:\n{names(score.get('unsupported_categories', []))}\n\n"
            f"Compound concepts present in scene:\n{names(score.get('present_compound_concepts', []))}\n\n"
            f"Compound concepts mentioned:\n{names(score.get('mentioned_compound_concepts', []))}\n\n"
            f"Unsupported compound concepts:\n{names(score.get('unsupported_compound_concepts', []))}\n\n"
            f"Recognized category mentions counted in precision:\n{category_lines}\n\n"
            f"Recognized tile-specific mentions counted in precision:\n{specific_lines}\n\n"
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