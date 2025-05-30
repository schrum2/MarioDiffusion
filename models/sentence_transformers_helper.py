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

# Get embeddings for a batch of captions and optional negative captions
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




def get_embeddings_split(batch_size, tokenizer, model, captions=None, neg_captions=None, device='cpu', max_length=20):

    padding_length = max(max([s.count(".") for s in captions]) if captions else 1, 
                     max([s.count(".") for s in neg_captions]) if neg_captions else 1)
    if (padding_length>max_length):
        raise ValueError(f"Token sequence length {padding_length} exceeds specified length {max_length}.")


    empty_split = split_sentences([""] * batch_size, padding_length)
    embeddings = get_embeddings_from_split(empty_split, tokenizer, model, device)

    if(captions is not None):
        captions_split = split_sentences(captions, padding_length)
        caption_embeddings = get_embeddings_from_split(captions_split, tokenizer, model, device)
        embeddings = torch.cat((embeddings, caption_embeddings), dim=0)
    
    if(neg_captions is not None):
        neg_split = split_sentences(neg_captions, padding_length)
        neg_embeddings = get_embeddings_from_split(neg_split, tokenizer, model, device)
        embeddings = torch.cat((neg_embeddings, embeddings), dim=0)
    

    #We don't need to unsqueeze this, we have an array of (batch_size, padding_length, encoding_size) already

    return embeddings.to(device)


#This method takes a caption batch in list form, and outputs a 2d list where every caption has been split by period
def split_sentences(caption_array, padding_length=20):  
    split_caption_array = []

    #Padding happens here
    for caption in caption_array:
        split_caption = [s.strip() for s in caption.split(".") if s.strip()]
        #This is the token padding, we just use an empty string
        split_caption += [""] * (padding_length - len(split_caption))
        split_caption_array.append(split_caption)
        
    return split_caption_array


#Expects all split vectors to be the same length
def get_embeddings_from_split(caption_batch, tokenizer, model, device='cpu'):
    all_caption_encodings = []
    for caption_sequence in caption_batch:
        #Encode the sequence of split captions as if it was a batch, should now be a [maxlength, embeddingsize] tensor
        caption_sequence = encode(caption_sequence, tokenizer, model, device)
        
        #We don't reshape this to avoid having to unsqueeze it later
        all_caption_encodings.append(caption_sequence)
    
    all_caption_encodings = torch.stack(all_caption_encodings, dim=0)
    return all_caption_encodings
        
        

if __name__ == "__main__":
    cap = split_sentences(["Hello. My name is George. How. Are you doing. Today?", "I am doing. Just fine. Thanks."])
    model_url = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    model = AutoModel.from_pretrained(model_url, trust_remote_code=True).to(device)
    get_embeddings_from_split(cap, tokenizer, model, device)
