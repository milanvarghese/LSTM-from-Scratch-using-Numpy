import numpy as np
import os
from src.LSTM import LSTMLayer
from src.nn import FullyConnectedLayer
from src.Loss import CrossEntropy  # Loss functions

# -----------------------------
# 1. Load the Poem Dataset
# -----------------------------
data_file_path = os.path.join("data", "data.txt")
with open(data_file_path, "r") as file:
    poem_text = file.read()
print("Loaded Poem Dataset:")
print(poem_text)

# -----------------------------
# 2. Build Character-Level Vocabulary
# -----------------------------
chars = sorted(list(set(poem_text)))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# -----------------------------
# 3. Hyperparameters and Model Setup
# -----------------------------
sequence_length = 25  # Length of input sequences
hidden_dim = 128      # Size of LSTM hidden state
learning_rate = 0.01
epochs = 10

input_dim = vocab_size  # One-hot vector size equals vocab size

# Initialize LSTM layer and output layer (maps hidden state to vocabulary distribution)
lstm = LSTMLayer(input_dim, hidden_dim)
output_layer = FullyConnectedLayer(hidden_dim, vocab_size)
loss_fn = CrossEntropy()

# -----------------------------
# 4. Utility Functions
# -----------------------------
def one_hot(index, dim):
    """Return one-hot column vector for a given index."""
    vec = np.zeros((dim, 1))
    vec[index] = 1
    return vec

# -----------------------------
# 5. Prepare Training Sequences
# -----------------------------
input_sequences = []
target_sequences = []
for i in range(0, len(poem_text) - sequence_length):
    input_seq = poem_text[i:i+sequence_length]
    target_seq = poem_text[i+1:i+sequence_length+1]
    input_sequences.append([char_to_idx[ch] for ch in input_seq])
    target_sequences.append([char_to_idx[ch] for ch in target_seq])

input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)

# -----------------------------
# 6. Training Loop with BPTT
# -----------------------------
print("\nStarting training...\n")
for epoch in range(epochs):
    total_loss = 0
    # Process each training sequence one by one.
    for seq_idx in range(len(input_sequences)):
        input_indices = input_sequences[seq_idx]
        target_indices = target_sequences[seq_idx]

        # Initialize hidden and cell states
        h_state = np.zeros((hidden_dim, 1))
        c_state = np.zeros((hidden_dim, 1))
        
        # Lists to store forward pass values for backpropagation.
        lstm_caches = []    # stores a copy of lstm.cache for each time step
        h_states = []       # hidden states per time step
        output_probs = []   # output probabilities per time step

        # ----- Forward Pass -----
        for t in range(sequence_length):
            x_t = one_hot(input_indices[t], vocab_size)
            h_state, c_state = lstm.forward(x_t, h_state, c_state)
            # Store a copy of hidden state and cache for time step t.
            h_states.append(h_state.copy())
            lstm_caches.append(lstm.cache.copy())
            # Compute output layer forward pass.
            logits = output_layer.forward(h_state.T)  # shape: (1, vocab_size)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            output_probs.append(probs)
        
        outputs = np.vstack(output_probs)  # shape: (sequence_length, vocab_size)
        
        # Create one-hot targets for the sequence.
        target_one_hot = np.zeros((sequence_length, vocab_size))
        for t, target_idx in enumerate(target_indices):
            target_one_hot[t, target_idx] = 1
        
        # Compute loss.
        loss = loss_fn.eval(target_one_hot, outputs)
        total_loss += loss
        
        # ----- Backpropagation -----
        # First: Backpropagate through the output layer for each time step.
        d_hidden_list = []  # gradients with respect to hidden state from output layer
        # Accumulate gradients for the output layer parameters.
        grad_W_accum = np.zeros_like(output_layer.W)
        grad_b_accum = np.zeros_like(output_layer.b)
        
        for t in range(sequence_length):
            # Compute derivative of softmax cross-entropy:
            # For softmax + cross-entropy, gradient w.r.t. logits is (predicted - target)
            target_row = np.zeros((1, vocab_size))
            target_row[0, target_indices[t]] = 1
            d_logits = output_probs[t] - target_row  # shape: (1, vocab_size)
            # Backprop through output layer.
            d_hidden = output_layer.backward(d_logits)  # shape: (1, hidden_dim)
            # Store d_hidden transposed to match hidden state shape (hidden_dim, 1).
            d_hidden_list.append(d_hidden.T)
            # Accumulate gradients for output layer.
            # The forward input to the output layer was h_states[t].T (shape: (1, hidden_dim)).
            grad_W_accum += np.dot(h_states[t], d_logits)  # (hidden_dim, vocab_size)
            grad_b_accum += d_logits  # (1, vocab_size)
        
        # Update output layer parameters (averaging gradients over time steps).
        output_layer.W -= learning_rate * (grad_W_accum / sequence_length)
        output_layer.b -= learning_rate * (grad_b_accum / sequence_length)
        
        # Second: Backpropagate through the LSTM in reverse time order.
        # d_h_next and d_c_next are gradients flowing from future time steps.
        d_h_next = np.zeros((hidden_dim, 1))
        d_c_next = np.zeros((hidden_dim, 1))
        for t in reversed(range(sequence_length)):
            # Before calling backward, restore the cache for time step t.
            lstm.cache = lstm_caches[t]
            # Total gradient for current hidden state: from output layer plus gradient from future.
            d_h_total = d_hidden_list[t] + d_h_next
            # Call backward on the LSTM cell.
            d_prev_h, d_x, d_prev_c = lstm.backward(d_h_total, d_c_next)
            d_h_next = d_prev_h  # gradient to pass to previous time step (for h)
            d_c_next = d_prev_c  # gradient to pass to previous time step (for c)
            # (d_x could be used if an embedding layer is present)
    
    avg_loss = total_loss / len(input_sequences)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# -----------------------------
# 7. Text Generation Function
# -----------------------------
def generate_text(seed_text, length=100):
    generated_text = seed_text
    h_state = np.zeros((hidden_dim, 1))
    c_state = np.zeros((hidden_dim, 1))
    
    # Feed seed text through the model.
    for ch in seed_text:
        x = one_hot(char_to_idx[ch], vocab_size)
        h_state, c_state = lstm.forward(x, h_state, c_state)
    
    current_char = seed_text[-1]
    for _ in range(length):
        x = one_hot(char_to_idx[current_char], vocab_size)
        h_state, c_state = lstm.forward(x, h_state, c_state)
        logits = output_layer.forward(h_state.T)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        next_idx = np.random.choice(range(vocab_size), p=probs.ravel())
        next_char = idx_to_char[next_idx]
        generated_text += next_char
        current_char = next_char
    return generated_text

# -----------------------------
# 8. Generate and Print Text
# -----------------------------
seed = "Roses are red,\n"
generated = generate_text(seed, length=200)
print("\nGenerated Text:\n")
print(generated)
