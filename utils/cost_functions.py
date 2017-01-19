import numpy as np
from utils.softmax import softmax
from utils.sigmoid import sigmoid
from hyperparams import num_noise, generator

def softmax_cost(center_word_vector, context_word_idx, output_vectors, center_word_index=None):
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

def negative_sampling_cost(center_word_vector, context_word_idx, output_vectors, center_word_index=None, num_noise=num_noise, noise_gen=generator):
    output_grad_at_context_word = np.zeros(output_vectors.shape)
    input_grad_at_context_word  = np.zeros(center_word_vector.shape)

    indices = [center_word_index]
    for k in range(num_noise):
        newidx, _ = noise_gen.sample()
        while newidx == center_word_index:
            newidx, _ = noise_gen.sample()
        indices += [newidx]

    labels = np.array([1] + [-1 for k in range(num_noise)])
    vecs = output_vectors[indices, :]

    probabilities = sigmoid(vecs.dot(center_word_vector) * labels)
    context_word_loss = -np.sum(np.log(probabilities))

    delta = labels * (probabilities - 1)
    input_grad_at_context_word = delta.reshape((1,num_noise + 1)).dot(vecs).flatten()

    gradtemp = delta.reshape((num_noise + 1, 1)).dot(center_word_vector.reshape(
        (1, center_word_vector.shape[0])))
    for k in range(num_noise + 1):
        output_grad_at_context_word[indices[k]] += gradtemp[k,:]

    return context_word_loss, input_grad_at_context_word, output_grad_at_context_word
