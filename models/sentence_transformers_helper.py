from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts, tokenizer, model, device='cpu'):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input.to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    embeddings = embeddings.to(device)

    return embeddings


def get_embeddings(batch_size, tokenizer, model, captions=None, neg_captions=None, device='cpu'):
    embeddings = encode([""]*batch_size, tokenizer, model, device)

    if captions is not None:
        caption_embeddings = encode(captions, tokenizer, model, device)
        embeddings = torch.cat((embeddings, caption_embeddings), dim=0)

    if neg_captions is not None:
        neg_embeddings = encode(neg_captions, tokenizer, model, device)
        embeddings = torch.cat((neg_embeddings, embeddings), dim=0)
    
    
    embeddings = embeddings.unsqueeze(1)
    
    return embeddings