import argparse
import json
import os

import torch
import util.common_settings as common_settings
from captions.util import extract_tileset

from models.general_training_helper import get_scene_from_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a text or JSON level against block embeddings.")
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="SMB1-block2vec-embeddings",
        help="Folder containing embeddings.pt",
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default=None,
        help="Path to a .txt level or JSON scene.",
    )
    parser.add_argument(
        "--tileset",
        type=str,
        default=common_settings.MARIO_TILESET,
        help="Tileset JSON used to map chars<->tile IDs for .txt levels.",
    )
    return parser.parse_args()


def load_scene_data(scene_path, char_to_id):
    if scene_path is None:
        raise ValueError("Provide --scene_path")

    _, extension = os.path.splitext(scene_path.lower())
    if extension == ".txt":
        with open(scene_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.rstrip("\n")]

        if not lines:
            raise ValueError("Input .txt scene is empty")

        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError("All rows in input .txt scene must have equal width")

        try:
            return [[char_to_id[ch] for ch in line] for line in lines]
        except KeyError as error:
            missing = str(error).strip("'")
            raise ValueError(f"Unknown tile char '{missing}'. Check --tileset mapping.") from error

    if extension == ".json":
        with open(scene_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "scene" in data:
            data = data["scene"]

        if not isinstance(data, list) or not data:
            raise ValueError("JSON scene must contain a level or a list of levels")

        return data

    raise ValueError("Scene file must be .txt or .json")


def validate_scene(scene, num_tiles):
    if not scene or not scene[0]:
        raise ValueError("Scene must not be empty")

    width = len(scene[0])
    for row in scene:
        if len(row) != width:
            raise ValueError("All scene rows must have the same width")
        for tile_id in row:
            if tile_id < 0 or tile_id >= num_tiles:
                raise ValueError(f"Tile ID {tile_id} is out of range for embeddings with {num_tiles} tiles")


def save_txt_scene(scene_ids, scene_txt_out, id_to_char):
    lines = ["".join(id_to_char[tile_id] for tile_id in row) for row in scene_ids]
    with open(scene_txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def is_scene_grid(data):
    return isinstance(data, list) and data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], int)


def is_scene_dataset(data):
    return (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and data[0]
        and isinstance(data[0][0], list)
    )


def reconstruct_scene(scene, embeddings):
    validate_scene(scene, embeddings.shape[0])

    tile_ids = torch.tensor(scene, dtype=torch.long).unsqueeze(0)
    latent_image = embeddings[tile_ids].permute(0, 3, 1, 2).float()

    output = get_scene_from_embeddings(latent_image, embeddings)
    reconstructed = output.argmax(dim=1)
    accuracy = (reconstructed == tile_ids).float().mean().item()

    return reconstructed[0].tolist(), accuracy, latent_image, output


def main():
    args = parse_args()

    emb_path = os.path.join(args.emb_dir, "embeddings.pt")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Could not find embeddings at: {emb_path}")

    embeddings = torch.load(emb_path, map_location="cpu")

    _, _, char_to_id, _ = extract_tileset(args.tileset)

    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    data = load_scene_data(args.scene_path, char_to_id)

    if is_scene_dataset(data):
        print(f"Loaded dataset with {len(data)} levels")
        accuracies = []
        for index, scene in enumerate(data):
            reconstructed_scene, accuracy, latent_image, output = reconstruct_scene(scene, embeddings)
            accuracies.append(accuracy)
            print(f"Level {index}: size {len(scene)}x{len(scene[0])}, accuracy {accuracy:.4f}")
            print(f"  Input latent shape: {tuple(latent_image.shape)}")
            print(f"  Output shape: {tuple(output.shape)}")
            #print(f"  Original scene: {json.dumps(scene)}")
            #print(f"  Reconstructed scene: {json.dumps(reconstructed_scene)}")

        print(f"Average reconstruction accuracy: {sum(accuracies) / len(accuracies):.4f}")
        return

    if not is_scene_grid(data):
        raise ValueError("JSON scene must be either a 2D level grid or a list of 2D levels")

    reconstructed_scene, accuracy, latent_image, output = reconstruct_scene(data, embeddings)

    print(f"Input scene size: {len(data)}x{len(data[0])}")
    print(f"Input latent shape: {tuple(latent_image.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Exact tile reconstruction accuracy: {accuracy:.4f}")
    #print("Original scene:")
    #print(json.dumps(data))
    #print("Reconstructed scene:")
    #print(json.dumps(reconstructed_scene))


if __name__ == "__main__":
    main()
