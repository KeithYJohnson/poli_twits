from hyperparams import num_features
from utils.softmax import softmax
import numpy as np

def skipgram(center_word_index, context_word_indices, input_vectors, output_vectors):
    center_word_vector = input_vectors[center_word_index]

    input_grad  = np.zeros(input_vectors.shape)
    output_grad = np.zeros(output_vectors.shape)
    loss = 0

    for context_word_idx in context_word_indices:
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

        loss        += context_word_loss
        input_grad  += input_grad_at_context_word
        output_grad += output_grad_at_context_word


    return loss, input_grad, output_grad
