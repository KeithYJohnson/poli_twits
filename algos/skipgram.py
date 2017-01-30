from hyperparams import num_features
import numpy as np
from utils.cost_functions import softmax_cost, negative_sampling_cost
from utils.arrange_vectors import split_vecs, stack_vecs

def skipgram(center_word_index, context_word_indices, word_vectors, cost_fn=negative_sampling_cost, hparams={}):
    center_word_vector = word_vectors[center_word_index]
    input_vectors, output_vectors = split_vecs(word_vectors)
    input_grad  = np.zeros(input_vectors.shape)
    output_grad = np.zeros(output_vectors.shape)
    loss = 0

    for context_word_idx in context_word_indices:
        [context_word_loss,
         input_grad_at_context_word,
         output_grad_at_context_word
        ] = cost_fn(center_word_vector, center_word_index, output_vectors, hparams=hparams)

        loss                             += context_word_loss
        input_grad[center_word_index, :] += input_grad_at_context_word
        output_grad                      += output_grad_at_context_word

        grad = stack_vecs(input_grad, output_grad)

    return loss, grad
