import numpy as np

class MovingWindowSplitter:
    """
    Custom splitter for time series cross-validation.

    This splitter generates indices to split data into training and test sets for time series data. It uses a moving window approach where both the training and test sets move through the data, with a specified gap between them.

    Parameters
    ----------
    n_splits : int, optional, default=None
        Number of splits. If specified, the train_size and step are ignored, and the splits are automatically calculated.
        
    train_size : float or int, default=0.2
        Size of the training set. If float, it represents the proportion of the dataset to include in the training set (between 0 and 1).
        
    test_size : float or int, default=0.1
        Size of the test set. If float, it represents the proportion of the dataset to include in the test set (between 0 and 1).
        
    step : int, optional, default=None
        Step size between each training set. If None, defaults to train_size.
        
    gap : int, default=0
        Number of samples to exclude between the training and test sets to avoid leakage.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import MovingWindowSplitter
    >>> X = np.arange(12).reshape(-1, 1)
    >>> splitter = MovingWindowSplitter(train_size=0.5, test_size=0.2)
    >>> for train_index, test_index in splitter.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [0 1 2 3 4] TEST: [5 6]
    TRAIN: [5 6 7 8 9] TEST: [10 11]
    """

    def __init__(self, n_splits=None, train_size=0.2, test_size=0.1, step=None, gap=0):
        """
        Initialize the MovingWindowSplitter.

        If n_splits is specified, then train_size and step are ignored.

        Parameters
        ----------
        n_splits : int, optional, default=None
            Number of splits. If specified, train_size, test_size, and step are ignored.
            
        train_size : float or int, default=0.2
            Size of the training set. If float, represents the proportion of the dataset to include in the training set (between 0 and 1).
            
        test_size : float or int, default=0.1
            Size of the test set. If float, represents the proportion of the dataset to include in the test set (between 0 and 1).
            
        step : int, optional, default=None
            Step size between each training set. Defaults to train_size if not specified.
            
        gap : int, default=0
            Number of samples to exclude between the training and test sets.
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        if step:
            step = step if step > 0 else 1
        self.step = step
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to split. The first dimension is the number of samples.
            
        y : array-like, optional
            Ignored, exists for compatibility.
            
        groups : array-like, optional
            Ignored, exists for compatibility.

        Yields
        ------
        train_index : ndarray
            The training set indices for that split.
            
        test_index : ndarray
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If the number of splits is greater than or equal to the number of samples in the data.
        """
        n_samples = X.shape[0]

        if self.n_splits is not None and self.n_splits >= n_samples:
            raise ValueError("Number of splits should be less than the number of samples")

        if self.n_splits is not None:
            self.train_size = int(n_samples / float(self.n_splits + 1))
            self.step = self.train_size
            test_size = self.test_size

            if test_size is not None: 
                if (0 < test_size <= 1): 
                    test_size = int(test_size * n_samples)
                elif test_size <= 0:
                    test_size = self.train_size
            else: 
                test_size=self.train_size
            
            if test_size > self.train_size:
                self.test_size = self.train_size
            else: 
                add_factor = int((n_samples - (self.train_size * self.n_splits)) / float(self.n_splits))
                self.train_size += add_factor
                self.step = self.train_size 
                self.test_size = test_size 

        if 0 <= self.train_size <= 1:
            self.train_size = int(self.train_size * n_samples)

        if 0 <= self.test_size <= 1:
            self.test_size = int(self.test_size * n_samples)

        if self.step is None:
            self.step = self.train_size

        if self.train_size + self.test_size >= n_samples:
            if self.n_splits is None:
                self.n_splits = 1
            self.test_size = n_samples - (self.train_size - self.gap)
            train_index = list(range(0, self.train_size))
            valid_index = list(range(self.train_size + self.gap, n_samples))
            yield np.array(train_index), np.array(valid_index)
        else:
            curr_ind = 0
            stop_loop = False
            count_splits = False
            if self.n_splits is None:
                count_splits = True
                self.n_splits = 0
            while not stop_loop:
                if count_splits:
                    self.n_splits += 1
                train_index = list(range(curr_ind, curr_ind + self.train_size))
                if curr_ind + self.train_size + self.test_size + self.gap >= n_samples or (n_samples - (curr_ind + self.train_size + self.test_size)) < self.test_size:
                    valid_index = list(range(n_samples - self.test_size, n_samples))
                    train_index = list(range(curr_ind, n_samples - self.test_size - self.gap))
                    stop_loop = True
                else:
                    valid_index = list(range(curr_ind + self.train_size + self.gap, curr_ind + self.train_size + self.gap + self.test_size))
                curr_ind += self.step
                yield np.array(train_index), np.array(valid_index)

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splits.

        Parameters
        ----------
        X : array-like, optional
            The input data, ignored in this method.
            
        y : array-like, optional
            The target variable, ignored in this method.
            
        groups : array-like, optional
            Group labels, ignored in this method.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return self.n_splits
