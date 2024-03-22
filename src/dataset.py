from numpy.typing import NDArray
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, raw_dataset: NDArray, shape: tuple, lengths: list):
        self.data = raw_dataset
        self.shape = shape
        self.lengths = lengths
        self.number_of_data = len(lengths)

    def get_data(self) -> NDArray:
        return self.data

    def get_partial_data(self, n: int):
        if n > self.number_of_data:
            raise Exception("Invalid number given")
        new_length = 0
        for i in range(n):
            new_length += self.lengths[i]
        
        return Dataset(raw_dataset=self.data[:new_length], shape=self.shape, 
                       lengths=self.lengths[:n])

    def get_lengths(self) -> list:
        return self.lengths

    def get_total_length(self) -> int:
        return sum(self.lengths)
    
    def get_reshaped_data(self) -> NDArray:
        return self.data.reshape((
            self.data.shape[0], self.data.shape[1], self.data.shape[2]
        ))

    def get_reshaped_data_fit(self) -> NDArray:
        return self.data.reshape((
            self.data.shape[0], self.data.shape[1], self.data.shape[2], 1
        ))

    def display_dataset(self, slice_index: int = 30):
        fig, ax = plt.subplots(nrows=1, ncols=self.number_of_data)

        start_index = 0
        for i in range(self.number_of_data):
            img = self.data[start_index:start_index+self.lengths[i]].T
            ax[i].imshow(img[:, :, slice_index])
            ax[i].title.set_text(f"data{i+1}")
            ax[i].grid(False)
            ax[i].axis('off')
            start_index += self.lengths[i]
        plt.show()
