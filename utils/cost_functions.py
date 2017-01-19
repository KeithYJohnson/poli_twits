import numpy as np
from utils.softmax import softmax

def softmax_cost(center_word_vector, context_word_idx, output_vectors):
    num_features = output_vectors.shape[1]

    probabilities = softmax(center_word_vector.dot(output_vectors.T))
    context_word_loss = -np.log(probabilities[context_word_idx])
    output_layer_error = probabilities
    output_layer_error[context_word_idx] -= 1

    num_predictions = output_layer_error.shape[0]

    input_grad_at_context_word = np.dot(
        output_layer_error.reshape(1, num_predictions),
        output_vectors
    ).flatten()

    output_grad_at_context_word = np.multiply(
        output_layer_error.reshape(num_predictions, 1),
        center_word_vector.reshape(1, num_features)
    )

    return context_word_loss, input_grad_at_context_word, output_grad_at_context_word
