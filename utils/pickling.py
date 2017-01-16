from hyperparams import *
from six.moves import cPickle as pickle

def create_pickle_filename():
    name = "numfeatures-{}--batch_size-{}--window_size-{}--numiters-{}--learning_rate-{}.pickle".format(num_features, batch_size, window_size, num_iterations, learning_rate)
    return name

def save_data(input_vectors, output_vectors):
    pickle_file = create_pickle_filename()
    try:
      f = open(pickle_file, 'wb')
      save = {
        'input_vectors':  input_vectors,
        'output_vectors': output_vectors,
        'word_vectors':   (input_vectors + output_vectors) / 2
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise



def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        input_vectors = save['input_vectors']
        output_vectors = save['output_vectors']
        word_vectors = save['word_vectors']
        del save  # hint to help gc free up memory

    return word_vectors, input_vectors, output_vectors
