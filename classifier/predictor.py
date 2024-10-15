import torch
from nets import GPTModel
from spam_classifier import spam_classifier

def load_model(model_path, config, num_classes, device):
    model = GPTModel(config)
    model.out_head = torch.nn.Linear(in_features=config["emb_dim"], out_features=num_classes)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model

def predict_spam(texts, model, tokenizer, device, max_length):
    predictions = {}
    for text in texts:
        predictions[text] = spam_classifier(text, model, tokenizer, device, max_length)
    return predictions