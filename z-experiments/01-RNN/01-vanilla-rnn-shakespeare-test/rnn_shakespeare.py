import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import sys
# New import for dynamic progress bar logging
from tqdm import tqdm
import time # <-- Added time module for profiling

# --- 0. Configuration and Device Setup ---

# Determine if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(f"--- Device Setup ---")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Note: On PyTorch, the 'CUDA_ERROR_INVALID_HANDLE' should be resolved by 
    # relying on its more flexible device management.

# --- Configuration (Kept consistent with TF version) ---
SEQ_LENGTH = 100       # How many previous characters the RNN looks at
RNN_UNITS = 512 * 2       # Size of the hidden state (memory capacity)
EMBEDDING_DIM = 256    # Dimension of the character embedding vectors
BATCH_SIZE = 1024 * 2
EPOCHS = 1000            # Start with 20; increase for better results
LEARNING_RATE = 0.001

# --- NEW CONFIG: Set to False to disable the per-batch progress bar ---
SHOW_BATCH_PROGRESS = False 
# ---------------------------------------------------------------------


# --- 1. Data Acquisition and Preprocessing ---

def load_text_data(filepath="shakespeare.txt"):
    """Loads text and creates character mappings."""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    
    # Simple check to download the file if not present (simulating tf.keras.utils.get_file)
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, filepath)
        except ImportError:
            # Fallback for systems that might not have urllib
            print("Error: Could not download file. Please download 'shakespeare.txt' manually.")
            sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    text_as_int = np.array([char2idx[c] for c in text])
    
    print(f"Total characters: {len(text)}")
    print(f"Vocabulary Size: {vocab_size}")
    
    return text_as_int, vocab_size, idx2char, char2idx


# --- 2. PyTorch Dataset and DataLoader ---

class ShakespeareDataset(Dataset):
    def __init__(self, text_as_int, seq_length):
        self.text_as_int = text_as_int
        self.seq_length = seq_length
        self.total_length = len(text_as_int)
        
        # Calculate how many full sequences we can extract
        self.num_sequences = (self.total_length - 1) // self.seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Calculate start index for the input sequence
        start_idx = idx * self.seq_length
        
        # Input: [char_t, char_t+1, ..., char_t+SEQ_LENGTH-1]
        input_sequence = self.text_as_int[start_idx:start_idx + self.seq_length]
        
        # Target: [char_t+1, char_t+2, ..., char_t+SEQ_LENGTH]
        # This is equivalent to shifting the input sequence by one
        target_sequence = self.text_as_int[start_idx + 1:start_idx + self.seq_length + 1]
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)
        
        return input_tensor, target_tensor


# --- 3. PyTorch Model Definition ---

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(RNNModel, self).__init__()
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. SimpleRNN layer (Pytorch's nn.RNN is the equivalent of Keras SimpleRNN)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            num_layers=1,
            batch_first=True  # Inputs will be (batch_size, seq_length, embedding_dim)
        )
        
        # 3. Dense (Linear) output layer
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, hidden_state=None):
        # 1. Embed the input: (B, L) -> (B, L, E)
        x = self.embedding(x)
        
        # 2. Pass through RNN: (B, L, E) -> (B, L, H)
        # output is (B, L, H), hidden_state is (num_layers, B, H)
        output, hidden_state = self.rnn(x, hidden_state)
        
        # 3. Pass all sequence outputs through the Dense layer
        # Output is (B*L, H) -> (B*L, V)
        # We flatten the batch and sequence dimensions for the linear layer
        output = self.fc(output.reshape(-1, output.shape[2]))
        
        return output, hidden_state

# --- 4. Training Loop ---

def train_model(model, dataloader, optimizer, loss_fn, epochs):
    print("\n--- 5. Starting Training (PyTorch RNN Sequence Learning) ---")
    model.train() # Set model to training mode
    
    total_training_start_time = time.time() # <-- Start total timer

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time() # <-- Start epoch timer
        total_loss = 0
        
        # Use tqdm to create a dynamic progress bar for the dataloader
        # The 'disable' parameter is set based on the global config flag
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", disable=not globals().get('SHOW_BATCH_PROGRESS', True))
        
        for data, target in loop: # Iterate over the loop object
            # Move data to the configured device (GPU/CPU)
            data, target = data.to(device), target.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            # output shape: (B*L, V), hidden_state shape: (1, B, H)
            output, _ = model(data)
            
            # Reshape target for loss calculation: (B, L) -> (B*L)
            target = target.reshape(-1)
            
            # Calculate loss
            loss = loss_fn(output, target)
            total_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            
            # Clip gradients to prevent exploding gradients (common in RNNs)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Only update the progress bar description if it is enabled
            if globals().get('SHOW_BATCH_PROGRESS', True):
                loop.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate the average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        
        epoch_end_time = time.time() # <-- End epoch timer
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Print the epoch summary only every 5 epochs (and the final one)
        if epoch % 5 == 0 or epoch == epochs:
            print(f'--- Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s ---') # <-- Updated printout
    
    total_training_end_time = time.time() # <-- End total timer
    total_duration = total_training_end_time - total_training_start_time
    print(f"\n--- Total Training Time: {total_duration:.2f} seconds ---") # <-- Print total time


# --- 5. Text Generation Function ---

def generate_text(model, start_string, num_generate, char2idx, idx2char):
    """Generates text using the trained model, step by step."""
    
    print("\n--- 6. Generating Text (Testing Long-Term Dependency) ---")
    model.eval() # Set model to evaluation mode
    
    # Convert start string to numbers and move to device
    # Ensure start string matches sequence length requirements
    padded_start_string = start_string[:SEQ_LENGTH].ljust(SEQ_LENGTH)
    input_eval = [char2idx[s] for s in padded_start_string]

    # Input tensor: (1, SEQ_LENGTH)
    input_tensor = torch.tensor([input_eval], dtype=torch.long).to(device) 
    
    text_generated = list(padded_start_string)
    temperature = 1.0
    
    # Initialize the hidden state to zero
    hidden_state = None
    
    # The generation process needs to iterate over the input sequence once 
    # to establish the initial context (hidden state).
    # We iterate over the input sequence to get the final hidden state
    with torch.no_grad():
        # First, process the input string to get the initial hidden state
        _, hidden_state = model(input_tensor, hidden_state)

        # The input for the first generation step should be the last character of the prompt
        # current_char_idx is the last character of the input tensor (1, 1)
        current_char_idx = input_tensor[:, -1].unsqueeze(1) 

        print(f"Start Sequence (used for context): '{padded_start_string.strip()}'")
        print("-" * 50)
        
        # Print the original prompt before the generated text
        # We start printing from the end of the original (unpadded) user input
        print(start_string.strip(), end='') 


        for i in range(num_generate):
            # Pass the single character input and the previous hidden state
            output, hidden_state = model(current_char_idx, hidden_state)

            # output is (1, vocab_size). Reshape to (vocab_size)
            logits = output.squeeze(0).squeeze(0) 

            # Apply temperature to logits for sampling
            logits = logits / temperature
            
            # Apply softmax to get probabilities
            probabilities = nn.functional.softmax(logits, dim=0)

            # Sample the next character index
            predicted_id = torch.multinomial(probabilities, num_samples=1).item()
            
            # Use the predicted ID as the next input
            current_char_idx = torch.tensor([[predicted_id]], dtype=torch.long).to(device)
            
            # Store and print the character immediately
            next_char = idx2char[predicted_id]
            sys.stdout.write(next_char)
            sys.stdout.flush() 

            text_generated.append(next_char)
    
    # Final newline after generation is complete
    print("\n" + "-" * 50)


# --- 6. Main Execution ---

if __name__ == '__main__':
    print("--- 1. Data Acquisition and Preprocessing ---")
    text_as_int, vocab_size, idx2char, char2idx = load_text_data()
    
    print("\n--- 2. PyTorch Dataset and DataLoader Setup ---")
    dataset = ShakespeareDataset(text_as_int, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    
    print("\n--- 3. Building PyTorch RNN Model ---")
    model = RNNModel(vocab_size, EMBEDDING_DIM, RNN_UNITS).to(device)
    
    # Loss function (CrossEntropyLoss includes softmax and handles integer targets)
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model summary (Standard PyTorch representation)
    print("\n--- BUILT MODEL SUMMARY (PyTorch Style) ---")
    print(model)
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 4. Training
    train_model(model, dataloader, optimizer, loss_fn, EPOCHS)

    # 5. Interactive Generation Loop
    
    print("\n--- 7. Starting Interactive Text Generation ---")
    print(f"Model trained. Enter a sequence of characters to continue the text.")
    print(f"Note: Input must be at least {SEQ_LENGTH} characters long (or shorter input will be padded with spaces).")
    
    while True:
        try:
            # Prompt user for input
            user_input = input("▶️ Start String (Ctrl+C to exit): ")
            
            if not user_input:
                # If user just presses Enter, use a random sample
                text_start_index = random.randint(1000, 100000)
                start_string_base = ''.join([idx2char[i] for i in text_as_int[text_start_index:text_start_index + SEQ_LENGTH]])
                # We only show the first 50 chars of the random sample
                print(f"-> Using random start sequence: '{start_string_base[:50]}...'")
                start_string = start_string_base
            else:
                start_string = user_input
                
            num_gen_input = input("How many characters to generate? [100]: ")
            num_generate = int(num_gen_input) if num_gen_input.isdigit() and int(num_gen_input) > 0 else 100
            
            # Generate new characters
            generate_text(model, start_string=start_string, num_generate=num_generate, char2idx=char2idx, idx2char=idx2char)
            
        except KeyboardInterrupt:
            print("\nExiting interactive generation.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    print("\n--- EXPERIMENT INSIGHTS ---")
    print("If the generated text loses narrative coherence quickly (e.g., changes subject or grammar after a few lines),")
    print("this is empirical evidence of the **Vanishing Gradient Problem** in the SimpleRNN.")
    print("To solve this in PyTorch, you would replace `nn.RNN` with `nn.LSTM` or `nn.GRU`.")