import argparse
import torch
from models.text_model import TransformerModel
from level_dataset import LevelDataset
from torch.utils.data import DataLoader

def masked_inputs(input_batch, tokenizer, device, mask_prob=0.15, generator=None):
    mask_token = tokenizer.token_to_id["[MASK]"]
    pad_token = tokenizer.token_to_id["[PAD]"]  # Don't mask [PAD] tokens
    input_batch = input_batch.to(device)

    rand_tensor = torch.rand(input_batch.shape, device=device, generator=generator)
    mask = (rand_tensor < mask_prob) & (input_batch != pad_token)
    input_batch[mask] = mask_token
    return input_batch

def evaluate_model(model, tokenizer, dataloader, device, mask_prob=0.15, console_output=True, eval_generator = None):

    if eval_generator == None:
        eval_generator = torch.Generator(device=device)
        eval_generator.manual_seed(0)  # Should this be a command line parameter? 

    model.eval()
    mask_token = tokenizer.token_to_id["[MASK]"]
    pad_token = tokenizer.token_to_id["[PAD]"]
    correct, total = 0, 0
    for batch in dataloader:
        for item in batch:
            masked_input = masked_inputs(item.clone(), tokenizer, device, mask_prob, generator=eval_generator)
            ground_truth = item.clone()
            masked_indices = (masked_input == mask_token).nonzero().squeeze(1)

            input_tensor = masked_input.unsqueeze(0).to(device)
            ground_truth_tensor = ground_truth.to(device)

            with torch.no_grad():
                output = model(input_tensor)
            
            predicted_ids = output[0].argmax(-1).tolist()
            
            for idx in masked_indices:
                predicted_token = tokenizer.id_to_token[predicted_ids[idx]]
                expected_token = tokenizer.id_to_token[ground_truth_tensor[idx].item()]
                
                if expected_token == "[PAD]":
                    continue # Don't investigate these

                try:
                    pad_index = ground_truth.tolist().index(pad_token) 
                except ValueError:
                    pad_index = len(ground_truth)

                if console_output:
                    print(f"Original: {(tokenizer.decode(ground_truth.tolist()[:pad_index]))}")
                    print(f"Masked  : {(tokenizer.decode(masked_input.tolist()[:pad_index]))}")
                    print(f"Predicted: {predicted_token} | Expected: {expected_token}\n")
                
                if predicted_token == expected_token:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Mask Prediction Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct} | Total: {total}")

    return (accuracy, correct, total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained transformer model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of captions to evaluate")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking each token")

    parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel.from_pretrained(args.model_path).to(device)
    print(f"Loaded model from {args.model_path}")
    
    dataset = LevelDataset(args.json, model.tokenizer, mode="text")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False) # No shuffle for post-eval
    
    if args.compare_checkpoints: # Evaluate all checkpoints and save a plot
        import os
        import re
        import json
        import matplotlib.pyplot as plt

        checkpoint_pattern = re.compile(r"checkpoint_epoch_(\d+)")
        checkpoint_info = []

        # Gather (epoch_num, path) pairs
        for subdir in os.listdir(args.model_path):
            match = checkpoint_pattern.match(subdir)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_path = os.path.join(args.model_path, subdir)
                checkpoint_info.append((epoch_num, checkpoint_path))

        # Sort by numeric epoch
        checkpoint_info.sort(key=lambda x: x[0])

        # Create deterministic generator
        eval_generator = torch.Generator(device=device)
        eval_generator.manual_seed(0) # Should this be a command line parameter? 

        # Evaluate and record stats
        log_entries = []
        epochs, accuracies = [], []

        for epoch_num, checkpoint_path in checkpoint_info:
            print(f"Evaluating checkpoint from epoch {epoch_num} at {checkpoint_path}")
            model = TransformerModel.from_pretrained(checkpoint_path).to(device)
            accuracy, correct, total = evaluate_model(
                model, model.tokenizer, dataloader, device, args.mask_prob,
                console_output=False, eval_generator=eval_generator
            )
            log_entry = {
                "epoch": epoch_num,
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            log_entries.append(log_entry)
            epochs.append(epoch_num)
            accuracies.append(accuracy)

        # Save plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, accuracies, marker='o')
        plt.title("Masked Token Prediction Accuracy by Checkpoint")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_checkpoint_accuracy_plot.png")
        plt.savefig(plot_path)
        print(f"Saved accuracy plot to {plot_path}")

        # Save JSONL
        jsonl_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_checkpoint_accuracy_log.jsonl")
        with open(jsonl_path, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved accuracy log to {jsonl_path}")

    else: # Just evaluate final and print results
        evaluate_model(model, model.tokenizer, dataloader, device, args.mask_prob)
