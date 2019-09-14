from keras.utils.data_utils import Sequence
import numpy as np

class BatchGenerator(Sequence):

    def __init__(self, h5_list, batch_size = 64):
        self.h5_list = h5_list
        self.h5_idx = 0
        # self.batch_idx = 0
        self.start_idx = 0
        self.batch_size = batch_size
        # get the lengths of the data sets
        h5_sizes = []
        for h5f in h5_list:
            h5_sizes.append(len(h5f['input']))
        self.h5_sizes = h5_sizes
        self.total_size = sum(h5_sizes)
        self.switching = False
        self.length = int(float(self.total_size)/batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # gets one batch
        total_start_idx = index*self.batch_size
        # decide where in this h5 file we are
        h5_idx = 0
        total_size_temp = 0
        for i in range(len(self.h5_sizes)):
            total_size_temp += self.h5_sizes[i]
            if total_start_idx > total_size_temp:
                h5_idx += 1
            else:
                break
        self.h5_idx = h5_idx

        start_idx = total_start_idx - sum(self.h5_sizes[:self.h5_idx])
        end_idx = start_idx + self.batch_size

        if end_idx > self.h5_sizes[self.h5_idx]:
            remainder_idx = end_idx - self.h5_sizes[self.h5_idx]
            end_idx = self.h5_sizes[self.h5_idx]
            self.switching = True
            next_h5_idx = self.h5_idx + 1
            if next_h5_idx == len(self.h5_sizes):
                next_h5_idx = 0

        oba_inputs = self.h5_list[self.h5_idx]['input'][start_idx:end_idx]
        oba_outputs = self.h5_list[self.h5_idx]['output'][start_idx:end_idx]

        if self.switching:
            # switching to next dataset
            oba_in_rem = self.h5_list[next_h5_idx]['input'][0:remainder_idx]
            oba_out_rem = self.h5_list[next_h5_idx]['output'][0:remainder_idx]

            oba_inputs = np.concatenate((oba_inputs, oba_in_rem))
            oba_outputs = np.concatenate((oba_outputs, oba_out_rem))
            self.h5_idx = next_h5_idx
            self.start_idx = remainder_idx
            self.switching = False
        else:
            self.start_idx = end_idx

        return oba_inputs, oba_outputs
        # yield(oba_inputs, oba_outputs)


    # def generator():
    #     while True:
    #         start_idx = self.start_idx
