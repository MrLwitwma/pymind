import pickle
import os

class Tokenizer():
    def __init__(self, lower = False):
        self.word_tokens = ['']
        self.lower = lower

    def fit(self, dataset: list[str]):
        for line in dataset:
            if self.lower:
                line.lower()

            words = line.split()
            for i in words:
                if i not in self.word_tokens:
                    self.word_tokens.append(i)

        return self.word_tokens

    def sentence_to_token(self, sentence :str):
        words = sentence.split(' ')
        tokens = []

        for word in words:
            if word in self.word_tokens:
                tokens.append(self.word_tokens.index(word))
            else:
                self.word_tokens.append(word)
                tokens.append(self.word_tokens.index(word))

        return tokens
    
    def tokens_to_sentence(self, tokens :list):
        sentence = ''

        for token in tokens:
            try:
                sentence += self.word_tokens[token] + ' '
            except IndexError:
                sentence += ' \0 '
        
        return sentence
    
    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "tokenizer.pik")

        with open(file_path, 'wb') as file:
            pickle.dump({
                'word_tokens': self.word_tokens
            }, file)

        return True
    
    def load(self, folder_path):
        with open(folder_path, 'rb') as file:
            data = pickle.load(file)
            self.word_tokens = data['word_tokens']
        return self.word_tokens

def pad_sequence(sequences, maxlen, padding):
    return_sequences = []
    for line in sequences:
        if len(line) < maxlen:
            if padding == 'post':
                for i in range(maxlen - len(line)):
                    line.append(0)
            elif padding == 'pre':
                tokens = []
                for i in range(maxlen - len(line)):
                    tokens.append(0)
                for j in line:
                    tokens.append(j)
                line = tokens
        return_sequences.append(line)
    return return_sequences

def dataset_splitter(dataset :list, maxsequence):
    '''
    Splits the data into two parts training data and predicting data.
    The last column is taken as the predicting column.

    Note: The all lists inside the list must be of equal length

    Example Usage:
        ```
        data = [[1, 2, 3], [1, 4, 5]]
        training, predicting = dataset_splitter(data, 3)
        print(training)
        # [[1, 2], [1, 4]]
        print(predicting)
        # [3, 5]
        ```
    '''
    training_df =[]
    predicting_df = []
    for sequence in dataset:
        training = []
        for i, seq in enumerate(sequence):
            if i == maxsequence - 1:
                predicting_df.append(seq)
            else:
                training.append(seq)
        training_df.append(training)
    return training_df, predicting_df





class LinearModel:
    def __init__(self):
        """
        Initializes a linear model.

        Attributes:
            m (float): Slope of the line.
            c (float): Intercept of the line.

        Uses the basic formula for linear equation:
            y = mx + c
        """
        self.m = 0  # slope
        self.c = 0  # intercept

    def fit(self, x :list[int], y:list[int]):
        """
        Calculate the slope (m) and intercept (c) using simple linear regression
        For simplicity, using a basic formula: 
            m = (n * Σxy - Σx * Σy) / (n * Σx² - (Σx)²) 
            c = (Σy - m * Σx) / n

        Example Usage:
            ```
                x = [1, 2, 4, 5]
                y = [2, 4, 8, 10]
                model.fit(x, y)

                model.predict([22]) -> 44.0
            ```
        """
        n = len(x)
        [1, 2, 3]
        xy_sum = sum(xi * yi for xi, yi in zip(x, y))
        x_sum = sum(x)
        y_sum = sum(y)
        x_squared_sum = sum(xi ** 2 for xi in x)

        self.m = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum ** 2)
        self.c = (y_sum - self.m * x_sum) / n

    def predict(self, x):
        """
        Return the predicted value of y = mx + c for given x        
        """
        return [self.m * xi + self.c for xi in x]


class LinearStackModel:
    """
    A class to manage a stack of linear models.

    This class allows training and predicting with multiple linear models, 
    where each model corresponds to a separate dataset or task.

    Attributes:
        models (list): A list of LinearModel instances.

    Example Usage:
        ```
            x_train = [
                [1, 2, 3],  # Dataset 1
                [4, 5, 6],  # Dataset 2
                [7, 8, 9],  # Dataset 3
            ]
            y_train = [
                [10, 20, 30],  # Target 1
                [40, 50, 60],  # Target 2
                [70, 80, 90],  # Target 3
            ]
            
            model = LinearStackModel(3)
            model.fit(x_train, y_train)

            # Predict using the models
            x_test = [1, 2, 3]  # Example features for prediction
            predictions = model.predict(x_test)
        ```
    """

    def __init__(self, num_models):
        """
        Initializes the LinearStackModel with the specified number of linear models.

        Args:
            num_models (int): The number of LinearModel instances to create.
        """
        self.models = [LinearModel() for _ in range(num_models)]

    def fit(self, x, y):
        """
        Trains each linear model on its respective dataset.

        Args:
            x (list of list): A list of input datasets, where each sublist corresponds to a model.
            y (list of list): A list of target datasets, where each sublist corresponds to a model.

        Returns:
            list: The trained LinearModel instances.
        """
        for model, x_i, y_i in zip(self.models, x, y):
            model.fit(x_i, [y_i])
        return self.models

    def predict(self, x):
        """
        Predicts using each linear model.

        Args:
            x (list): A single input dataset to be used by all models.

        Returns:
            list: A list of predictions from each model.
        """
        return [model.predict(x) for model in self.models]





class RNN:
    def __init__(self):
        pass
    def fit(self):
        pass
    def predict(self):
        pass