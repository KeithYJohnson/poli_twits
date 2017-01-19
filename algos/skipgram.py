from hyperparams import num_features
import numpy as np
from utils.cost_functions import softmax_cost, negative_sampling_cost

def skipgram(center_word_index, context_word_indices, input_vectors, output_vectors, cost_fn=negative_sampling_cost):
    center_word_vector = input_vectors[center_word_index]

    input_grad  = np.zeros(input_vectors.shape)
    output_grad = np.zeros(output_vectors.shape)
    loss = 0

    for context_word_idx in context_word_indices:
        [context_word_loss,
         input_grad_at_context_word,
         output_grad_at_context_word
        ] = cost_fn(center_word_vector, context_word_idx, output_vectors, center_word_index=center_word_index)

        loss        += context_word_loss
        input_grad  += input_grad_at_context_word
        output_grad += output_grad_at_context_word


    return loss, input_grad, output_grad
