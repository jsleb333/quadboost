import json
import numpy as np


class LabelEncoder:
    """
    Class that encodes and decodes labels according to a given encoding.
    """
    def __init__(self, labels_encoding):
        """
        labels_encoding (Dictionary): Dictionary of {label:encoding} where the encodings are arrays of -1, 0 or 1.
        """
        self.labels_encoding = labels_encoding
        self.labels = sorted([label for label in self.labels_encoding])

        self.labels_to_idx = {label:idx for idx, label in enumerate(self.labels)}
        
        self.n_classes = len(self.labels)
        self.encoding_dim = len(self.labels_encoding[self.labels[0]])

        self.encoding_matrix = np.array([self.labels_encoding[label] for label in self.labels])
        self.weights_matrix = np.abs(self.encoding_matrix)/np.sum(np.abs(self.encoding_matrix), axis=1).reshape(-1,1)


    def encode_labels(self, Y):
        """
        Y (Iterable of length 'n_examples'): Labels of the examples.

        Returns the encoded labels as an array of shape (n_examples, encoding_dim) and the associated encoding weights as an array of the same shape.
        """
        encoded_Y = np.zeros((len(Y), self.encoding_dim))
        weights = np.zeros_like(encoded_Y)
        for i, label in enumerate(Y):
            label_idx = self.labels_to_idx[label]

            encoded_Y[i] = self.encoding_matrix[label_idx]
            weights[i] = self.weights_matrix[label_idx]
        
        return encoded_Y, weights


    def decode_labels(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim)): Array of encoded labels. It can contain real numbers, for instance in the case where encoded_Y are predictions.

        Decodes the labels by computing a score between encoded labels and the encoding matrix and taking the argmax as the class.

        Returns a list of decoded labels.
        """
        n_examples = encoded_Y.shape[0]
        scored_Y = self.encoding_score(encoded_Y) # Shape: (n_examples, n_classes)
        decoded_Y_idx = np.argmax(scored_Y, axis=1) # Shape: (n_examples,)
        decoded_Y = []
        for idx in decoded_Y_idx:
            decoded_Y.append(self.labels[idx])

        return decoded_Y


    def encoding_score(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim)): Array of encoded labels. It can contain real numbers, for instance in the case where encoded_Y are predictions.

        By default, computes the scalar product of the encoded labels with each encoding from the weighted encoding matrix.
        For one-hot vectors encoding, it is equivalent to do nothing since the encoding matrix is the identity.

        TODO implement quadratic score
        """
        weighted_encoding = self.encoding_matrix * self.weights_matrix
        score = encoded_Y.dot(weighted_encoding.T)
        return score


    @staticmethod
    def load_encodings(encoding_name, filepath='./encodings/encodings.json', convert_to_int=False):
        """
        Returns a LabelEncoder objects from the encodings in the loaded file.

        Encodings can be written in twos ways: Verbally or not.
        If verbal, the encodings should be a JSON dictionary of
            {"Description of the encoded feature": [[labels that have this feature], [labels that do not care about this feature]]}
        if not, the encodings should be a JSON dictionary of
            {label:[list of 1, 0 or -1 that encodes the label]}
        """
        with open(filepath) as file:
            encodings_settings = json.load(file)[encoding_name]
            verbal = encodings_settings['verbal']
            encodings = encodings_settings['encoding']

        if verbal:
            verbal_encodings = encodings
            labels = sorted(set(l for f, [ones, zeros] in verbal_encodings.items() for l in ones+zeros))
            encodings = {l:-np.ones(len(verbal_encodings)) for l in labels}
            for i, (feature, [ones, zeros]) in enumerate(verbal_encodings.items()):
                for label in ones:
                    encodings[label][i] = 1
                for label in zeros:
                    encodings[label][i] = 0
        if convert_to_int:
            encodings = {int(label):encoding for label, encoding in encodings.items()}

        return LabelEncoder(encodings)


class OneHotEncoder(LabelEncoder):
    def __init__(self, Y):
        labels = sorted(set(Y))
        one_hot_encoding = lambda idx: 2*np.eye(1, len(labels), k=idx)[0]-1
        labels_encoding = {label:one_hot_encoding(idx) for idx, label in enumerate(labels)}

        super().__init__(labels_encoding)


class AllPairsEncoder(LabelEncoder):
    def __init__(self, Y):
        labels = sorted(set(Y))
        n_classes = len(labels)
        encoding_dim = int(n_classes*(n_classes-1)/2)
        labels_encoding = {label:np.zeros(encoding_dim) for label in labels}
        idx_to_pairs = [(i,j) for i in range(n_classes) for j in range(i+1, n_classes)]
        
        for idx, (i, j) in enumerate(idx_to_pairs):
            labels_encoding[labels[i]][idx] = 1
            labels_encoding[labels[j]][idx] = -1

        super().__init__(labels_encoding)
        