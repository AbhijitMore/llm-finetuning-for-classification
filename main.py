import time
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from nets.model_configs import BASE_CONFIG, CHOOSE_MODEL, load_weights_into_gpt, download_and_load_gpt2
from nets.nets import GPTModel
from utils.data_preprocessing import create_balanced_dataset, random_split
from utils.dataset import SpamDataset
from utils.utils import get_tokenizer, calc_loss_loader, calc_accuracy_loader, calc_loss_batch, evaluate_model, plot_values
from classifier.predictor import load_model, predict_spam


# Define the main training function
def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # Lists to store training/validation losses and accuracies
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    print("\nStarting training process...\n")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        # Initialize tqdm for the training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True, total=len(train_loader))
        epoch_train_loss = 0

        for input_batch, target_batch in pbar:
            optimizer.zero_grad()  # Reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate loss
            optimizer.step()  # Update model weights
            
            epoch_train_loss += loss.item()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Update tqdm description with the latest loss values
                pbar.set_postfix({'Train Loss': f'{train_loss:.3f}', 'Val Loss': f'{val_loss:.3f}'})

            # Update progress bar
            pbar.update(1)

        # Average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}: Average Training Loss: {avg_train_loss:.4f}")

        # Close the progress bar after each epoch
        pbar.close()

        # Calculate and print accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        # Store and print epoch stats
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}: Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%\n")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# Define the test function
def test(model, test_loader, device):
    print("\nStarting model evaluation on the test set...")
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")
    return test_accuracy

# Main function
def main():
    
    SEED = 42
    torch.manual_seed(SEED)

    # url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    # zip_path = "sms_spam_collection.zip"
    # extracted_path = "sms_spam_collection"
    data_file_path = "sms_spam_collection/SMSSpamCollection.tsv"

    # df = download_and_prepare_data(url, zip_path, extracted_path, data_file_path)
    print("\nLoading dataset...\n")

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    print("Tokenizing dataset...\n")
    tokenizer = get_tokenizer()

    train_dataset = SpamDataset(dataset = train_df, max_length=None, tokenizer=tokenizer)
    max_length = train_dataset.max_length
    val_dataset = SpamDataset(dataset = validation_df,max_length=max_length,tokenizer=tokenizer)
    test_dataset = SpamDataset(dataset = test_df,max_length=max_length,tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size, num_workers=num_workers, drop_last=False,)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False,)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading GPT2 model...\n")
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    num_classes=2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    model.to(device)
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    eval_freq = 50
    eval_iter = 5

    # Start training
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs, examples_seen = train(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.\n")

    # Plot loss and accuracy values
    plot_values(num_epochs, examples_seen, train_losses, val_losses, label="loss")
    plot_values(num_epochs, examples_seen, train_accs, val_accs, label="accuracy")

    # Test the model
    test_accuracy = test(model, test_loader, device)

    model_path = 'review_classifier.pth'
    #save the
    print(f"Saving the trained model to {model_path}...\n")
    torch.save(model.state_dict(), model_path)

    # # Uncomment if model is not saved
    # model = load_model(model_path, BASE_CONFIG, num_classes, device)

    # # Uncomment if you want to see predictions on review examples
    # texts = [
    # "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    # "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    # ]

    # # Make predictions
    # predictions = predict_spam(texts, model, tokenizer, device, max_length)

    # # Output the predictions
    # for text, result in predictions.items():
    #     print(f"Text: {text}\nPrediction: {result}\n")


# Run the main function
if __name__ == "__main__":
    main()