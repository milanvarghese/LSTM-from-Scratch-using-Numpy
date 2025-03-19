import numpy as np
import os
from src.LSTM import LSTMLayer
from src.nn import FullyConnectedLayer
from src.Activation import softmax
from src.Loss import CrossEntropy  # Loss functions
from src.perplexity import perplexity
from src.basic import BasicTokenizer
import pandas as pd


def sample_top_k(probs, k=5):
    """Select one of the top-k probable indices randomly based on their probabilities."""
    # Get the indices of the top-k probabilities
    top_k_indices = np.argsort(probs)[-k:]  # Sort and pick last k (highest)
    
    # Normalize probabilities for top-k
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()  # Convert to valid probability distribution
    
    # Sample from the top-k tokens
    chosen_idx = np.random.choice(top_k_indices, p=top_k_probs)
    
    return chosen_idx


# -----------------------------
# 1. Load the Poem Dataset
# -----------------------------
data_file_path = os.path.join("data", "got.txt")
with open(data_file_path, "r") as file:
    poem_text = file.read()
    poem_text = poem_text.split("\n")
    poem_text = " ".join(poem_text)

tokenizer = BasicTokenizer()
tokenizer.train(poem_text, 256 + 600) # 256 are the byte tokens, then do 200 merges
tokens = tokenizer.encode(poem_text)
# -----------------------------
# 2. Build Character-Level Vocabulary
# -----------------------------
# words = poem_text.split(" ")
# s = set(words)
# chars = sorted(list(tokens))
vocab_size = len(set(tokens))
vocab = list(set(tokens))
# vocab = {}
# for tok in tokens:
#     vocab[tok] = tokenizer.decode([tok])
print("Vocabulary size:", vocab_size)

# char_to_idx = {ch: i for i, ch in enumerate(chars)}
# idx_to_char = {i: ch for i, ch in enumerate(chars)}


# -----------------------------
# 3. Hyperparameters and Model Setup
# -----------------------------
sequence_length = 5  # Length of input sequences
hidden_dim = 512      # Size of LSTM hidden state
learning_rate = 0.01
epochs = 50
batch_size = 15

params_str = f"sq_{sequence_length}-hd_{hidden_dim}-lr_{learning_rate}-ep_{epochs}"

input_dim = vocab_size  # One-hot vector size equals vocab size

# Initialize LSTM layer and output layer (maps hidden state to vocabulary distribution)
lstm = LSTMLayer(input_dim, hidden_dim)
output_layer = FullyConnectedLayer(hidden_dim, vocab_size)
sftmx = softmax()
loss_fn = CrossEntropy()
losses = []
perpelexities = []
# -----------------------------
# 4. Utility Functions
# -----------------------------
def one_hot(index, dim):
    """Return one-hot column vector for a given index."""
    # if len(index) == 1:   # handle batching
    vec = np.zeros((dim, 1))
    vec[index] = 1
    return vec
    # else:
    #     vecs = []
    #     for idx in index:
    #         vec = np.zeros((dim,1))
    #         vec[idx] = 1
    #         vecs.append(vec)
    #     return np.array(vecs)
# def one_hot(indices, dim):
#     vec = np.zeros((len(indices), dim))  # Shape: (batch_size, dim)
#     vec[np.arange(len(indices)), indices] = 1  # Set ones at given indices
#     return vec

# -----------------------------
# 5. Prepare Training Sequences
# -----------------------------
input_sequences = []
target_sequences = []
# poem_text = words
for i in range(0, len(tokens) - sequence_length):
    
    input_seq = tokens[i:i+sequence_length]
    target_seq = tokens[i+1:i+sequence_length+1]
    # print(f"input seq: {tokenizer.decode(input_seq)}\n target seq: {tokenizer.decode(target_seq)}")
    # exit()
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)




# print("before batching: ",input_sequences.shape)
# input_sequences = input_sequences.reshape(int(len(input_sequences)/batch_size),batch_size , 50)
# target_sequences = input_sequences.reshape(int(len(target_sequences)/batch_size),batch_size , 50)
# print("after batching: ",input_sequences.shape)
# exit()



# -----------------------------
# 6. Training Loop with BPTT
# -----------------------------
print("\nStarting training...\n")
for epoch in range(epochs):
    total_loss = 0
    epoch_perpelexities = []
    # Process each training sequence one by one.
    
    for seq_idx in range(len(input_sequences)):
        input_tokens = input_sequences[seq_idx]
        target_tokens = target_sequences[seq_idx]
       

        # Initialize hidden and cell states
        h_state = np.zeros((hidden_dim, 1))
        c_state = np.zeros((hidden_dim, 1))
        
        # Lists to store forward pass values for backpropagation.
        lstm_caches = []    # stores a copy of lstm.cache for each time step
        h_states = []       # hidden states per time step
        output_probs = []   # output probabilities per time step

        # ----- Forward Pass -----
        for t in range(sequence_length):
            # print(f"input_indices[t]: {input_tokens[t]}\tt: {t}")
            # exit()
            x_t = one_hot(vocab.index(input_tokens[t]), vocab_size) # index of token in input_induce
            
            h_state, c_state = lstm.forward(x_t, h_state, c_state)
            # Store a copy of hidden state and cache for time step t.
            h_states.append(h_state.copy())
            lstm_caches.append(lstm.cache.copy())
            # Compute output layer forward pass.
            logits = output_layer.forward(h_state.T)  # shape: (1, vocab_size)
            probs = sftmx.forward(logits)
            
            
            # exp_logits = np.exp(logits - np.max(logits))
            # probs = exp_logits / np.sum(exp_logits)
            output_probs.append(probs)
        
        outputs = np.vstack(output_probs)  # shape: (sequence_length, vocab_size)
        
        # Create one-hot targets for the sequence.
        target_one_hot = np.zeros((sequence_length, vocab_size))
        for t, target_idx in enumerate(target_tokens):
            target_one_hot[t, vocab.index(target_idx)] = 1
        
        
        target_probs = [probs[0][vocab.index(i)] for i in target_tokens]
        perp = perplexity(probabilities=target_probs)
        epoch_perpelexities.append(perp)
        
        
        
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
            target_row[0, vocab.index(target_tokens[t])] = 1
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
    avg_perp = sum(epoch_perpelexities)/len(epoch_perpelexities)
    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}\t perplexity: {avg_perp:.4f}")
    losses.append(avg_loss)
    perpelexities.append(avg_perp)

# -----------------------------
# 7. Text Generation Function
# -----------------------------
def generate_text(seed_text, length=100, temp=0.7):
    # seed_text = seed_text.split(" ")
    generated_text = seed_text
    h_state = np.zeros((hidden_dim, 1))
    c_state = np.zeros((hidden_dim, 1))
    seed_text_tokens = tokenizer.encode(seed_text)
    generated_tokens = []
    # Feed seed text through the model.
    for tok in seed_text_tokens:
        x = one_hot(vocab.index(tok), vocab_size) #char_to_idx
        h_state, c_state = lstm.forward(x, h_state, c_state)
    
    current_token = seed_text_tokens[-1]
    for _ in range(length):
        x = one_hot(current_token, vocab_size)
        h_state, c_state = lstm.forward(x, h_state, c_state)
        logits = output_layer.forward(h_state.T)
        # exp_logits = np.exp(logits - np.max(logits))
        # probs = exp_logits / np.sum(exp_logits)
        probs = sftmx.forward(logits)
        # next_idx = np.random.choice(range(vocab_size), p=probs.ravel())
        next_idx = sample_top_k(probs=probs.flatten(), k=50)
        # next_idx = np.argmax(probs)
        generated_tokens.append(vocab[int(next_idx)])   
        # # next_idx = np.argmax(probs)
        # next_char = tokenizer.decode([int(next_idx)])
        # print(next_char)
        # generated_text += next_char
        # generated_text += " "
        current_token = next_idx
    return generated_tokens

losses_df =  pd.DataFrame({"losses": losses, "perpelexities": perpelexities})
losses_df.to_csv(f"/Users/hamza.mahmood/Developer/deep_learning_course/LSTM-from-Scratch-using-Numpy/results/{params_str}.csv", index_label=False)

# -----------------------------
# 8. Generate and Print Text
# -----------------------------
seed = "Hello worl"
generated = generate_text(seed, length=200)
# print(f"\nGenerated tokes: {generated}\ntext:\n")
print(tokenizer.decode(generated))


