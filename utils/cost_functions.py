import numpy as np

# A hack so when I run this file from the command line
# python knows where to find the utils module
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.softmax import softmax
from utils.sigmoid import sigmoid
from utils.normalize_rows import normalize_rows
from ipdb import set_trace as st

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

def negative_sampling_cost(center_word_vector, context_word_idx, output_vectors, ctx_indices=None, hparams={}):
    noise_gen = hparams.get('noise_gen')
    num_noise = hparams.get('num_noise')

    output_grad_at_context_word = np.zeros(output_vectors.shape)
    input_grad_at_context_word  = np.zeros(center_word_vector.shape)

    indices = [context_word_idx]
    if not ctx_indices:
        for k in range(num_noise):
            newidx, _ = noise_gen.sample()
            while newidx == context_word_idx:
                newidx, _ = noise_gen.sample()
            indices += [newidx]
    else:
        indices += ctx_indices

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

if __name__ == '__main__':
    import random
    rndstate = random.getstate()
    num_features = 10
    num_train    = 11
    cw_vec       = normalize_rows(np.random.randn(1, num_features))
    output_vecs  = normalize_rows(np.random.randn(num_train, num_features))
    # Stacking just so an np.nditer(stacked) iterates through all the parameters.
    stacked      = np.vstack([cw_vec, output_vecs])
    hparams      = { 'num_noise': 2 }
    ctx_indices  = [1, 3]
    center_word_idx = 2

    h = 1e-4

    loss, ingrad, outgrad = negative_sampling_cost(stacked[0,:], center_word_idx, stacked[1:,:], hparams=hparams, ctx_indices=ctx_indices)
    analytical_grad = np.vstack([ingrad, outgrad])
    numgrad = np.zeros(analytical_grad.shape)

    it = np.nditer(stacked, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index
        stacked[ix] += h
        random.setstate(rndstate)
        plus_loss, _, _ = negative_sampling_cost(stacked[0,:], center_word_idx, stacked[1:,:], hparams=hparams, ctx_indices=ctx_indices)
        stacked[ix] -= 2 * h # restore to previous value (very important!)
        random.setstate(rndstate)
        minus_loss, _, _ = negative_sampling_cost(stacked[0,:], center_word_idx, stacked[1:,:], hparams=hparams, ctx_indices=ctx_indices)
        numgrad_at_ix = (plus_loss - minus_loss) / (2* h);
        numgrad[ix] = numgrad_at_ix
        reldiff = abs(numgrad_at_ix - analytical_grad[ix]) / max(1, abs(numgrad_at_ix), abs(analytical_grad[ix]))

        if reldiff > 1e-4:
            print("Gradient check failed.  Reldiff: ", reldiff)
            print("First gradient error found at index %s" % str(ix))
            print("Analytical gradient {} \t Numerical gradient: {}".format(analytical_grad[ix], numgrad[ix]))

        it.iternext()
