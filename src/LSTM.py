import numpy as np
from .Layer import Layer
from .Activation import sigmoid, tanh

class LSTMLayer(Layer):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.input_dim = input_dimension
        self.hidden_dim = hidden_dimension
        
        # Initialize weights and biases for each gate with small random values.
        self.weight_forget_gate = np.random.randn(hidden_dimension, input_dimension + hidden_dimension) * 0.01
        self.bias_forget_gate = np.zeros((hidden_dimension, 1))
        
        self.weight_input_gate = np.random.randn(hidden_dimension, input_dimension + hidden_dimension) * 0.01
        self.bias_input_gate = np.zeros((hidden_dimension, 1))
        
        self.weight_candidate = np.random.randn(hidden_dimension, input_dimension + hidden_dimension) * 0.01
        self.bias_candidate = np.zeros((hidden_dimension, 1))
        
        self.weight_output_gate = np.random.randn(hidden_dimension, input_dimension + hidden_dimension) * 0.01
        self.bias_output_gate = np.zeros((hidden_dimension, 1))
        
        # Create instances of activation functions.
        self.sigmoid_activation = sigmoid()  # instance of sigmoid class
        self.tanh_activation = tanh()          # instance of tanh class

    def forward(self, current_input, prev_hidden_state, prev_cell_state):
        """
        Forward pass for one time step of the LSTM.
        
        Parameters:
          current_input    : Current input vector (shape: input_dim x 1)
          prev_hidden_state: Hidden state from previous time step (shape: hidden_dim x 1)
          prev_cell_state  : Cell state from previous time step (shape: hidden_dim x 1)
          
        Returns:
          current_hidden_state: Updated hidden state (shape: hidden_dim x 1)
          current_cell_state  : Updated cell state (shape: hidden_dim x 1)
        """
        # Combine previous hidden state and current input.
        combined_input = np.vstack((prev_hidden_state, current_input))
        
        # Forget gate activation.
        forget_gate = self.sigmoid_activation.forward(
            np.dot(self.weight_forget_gate, combined_input) + self.bias_forget_gate)
        
        # Input gate activation.
        input_gate = self.sigmoid_activation.forward(
            np.dot(self.weight_input_gate, combined_input) + self.bias_input_gate)
        
        # Candidate cell state activation.
        candidate_cell = self.tanh_activation.forward(
            np.dot(self.weight_candidate, combined_input) + self.bias_candidate)
        
        # Compute current cell state.
        current_cell_state = forget_gate * prev_cell_state + input_gate * candidate_cell
        
        # Output gate activation.
        output_gate = self.sigmoid_activation.forward(
            np.dot(self.weight_output_gate, combined_input) + self.bias_output_gate)
        
        # Compute current hidden state.
        current_hidden_state = output_gate * self.tanh_activation.forward(current_cell_state)
        
        # Save intermediate values for use in backpropagation.
        self.cache = {
            "combined_input": combined_input,
            "forget_gate": forget_gate,
            "input_gate": input_gate,
            "candidate_cell": candidate_cell,
            "prev_cell_state": prev_cell_state,
            "current_cell_state": current_cell_state,
            "output_gate": output_gate,
            "current_hidden_state": current_hidden_state
        }
        
        self.setPrevIn(current_input)
        self.setPrevOut(current_hidden_state)
        
        return current_hidden_state, current_cell_state

    def gradient(self):
        # For LSTM, use the specialized backward() method.
        raise NotImplementedError("LSTM layer does not use gradient(); call backward() instead.")

    def backward(self, d_current_hidden, d_cell_state_next):
        """
        Backward pass for one time step of the LSTM.
        
        Parameters:
          d_current_hidden : Gradient w.r.t. current hidden state (shape: hidden_dim x 1)
          d_cell_state_next: Gradient w.r.t. cell state from future time step (shape: hidden_dim x 1)
          
        Returns:
          d_prev_hidden_state: Gradient to be passed to the previous hidden state (h_{t-1})
          d_input            : Gradient to be passed to the current input (x_t)
          d_prev_cell_state  : Gradient to be passed to the previous cell state (c_{t-1})
        """
        # Retrieve cached values from forward pass.
        cache = self.cache
        combined_input = cache["combined_input"]
        forget_gate = cache["forget_gate"]
        input_gate = cache["input_gate"]
        candidate_cell = cache["candidate_cell"]
        prev_cell_state = cache["prev_cell_state"]
        current_cell_state = cache["current_cell_state"]
        output_gate = cache["output_gate"]
        current_hidden_state = cache["current_hidden_state"]
        
        # Compute derivative of tanh(current_cell_state).
        tanh_current_cell = np.tanh(current_cell_state)
        d_tanh_current_cell = 1 - tanh_current_cell ** 2
        
        # Total gradient on cell state (accumulating gradient flowing from current hidden state and future cell state).
        d_total_cell_state = d_cell_state_next + d_current_hidden * output_gate * d_tanh_current_cell
        
        # Gradients for output gate.
        d_output_gate = d_current_hidden * tanh_current_cell
        d_output_gate_pre = d_output_gate * output_gate * (1 - output_gate)
        
        # Gradients for input gate.
        d_input_gate = d_total_cell_state * candidate_cell
        d_input_gate_pre = d_input_gate * input_gate * (1 - input_gate)
        
        # Gradients for forget gate.
        d_forget_gate = d_total_cell_state * prev_cell_state
        d_forget_gate_pre = d_forget_gate * forget_gate * (1 - forget_gate)
        
        # Gradients for candidate cell state.
        d_candidate_cell = d_total_cell_state * input_gate
        d_candidate_pre = d_candidate_cell * (1 - candidate_cell ** 2)
        
        # Gradients with respect to weights and biases for each gate.
        grad_weight_forget_gate = np.dot(d_forget_gate_pre, combined_input.T)
        grad_bias_forget_gate = d_forget_gate_pre
        
        grad_weight_input_gate = np.dot(d_input_gate_pre, combined_input.T)
        grad_bias_input_gate = d_input_gate_pre
        
        grad_weight_candidate = np.dot(d_candidate_pre, combined_input.T)
        grad_bias_candidate = d_candidate_pre
        
        grad_weight_output_gate = np.dot(d_output_gate_pre, combined_input.T)
        grad_bias_output_gate = d_output_gate_pre
        
        # Store gradients for parameter updates.
        self.grad_weight_forget_gate = grad_weight_forget_gate
        self.grad_bias_forget_gate = grad_bias_forget_gate
        
        self.grad_weight_input_gate = grad_weight_input_gate
        self.grad_bias_input_gate = grad_bias_input_gate
        
        self.grad_weight_candidate = grad_weight_candidate
        self.grad_bias_candidate = grad_bias_candidate
        
        self.grad_weight_output_gate = grad_weight_output_gate
        self.grad_bias_output_gate = grad_bias_output_gate
        
        # Compute gradient with respect to the combined input.
        d_combined_input = (
            np.dot(self.weight_forget_gate.T, d_forget_gate_pre) +
            np.dot(self.weight_input_gate.T, d_input_gate_pre) +
            np.dot(self.weight_candidate.T, d_candidate_pre) +
            np.dot(self.weight_output_gate.T, d_output_gate_pre)
        )
        
        # Split gradient into components: one for previous hidden state and one for input.
        d_prev_hidden_state = d_combined_input[:self.hidden_dim, :]
        d_input = d_combined_input[self.hidden_dim:, :]
        
        # Gradient for previous cell state.
        d_prev_cell_state = d_total_cell_state * forget_gate
        
        return d_prev_hidden_state, d_input, d_prev_cell_state




