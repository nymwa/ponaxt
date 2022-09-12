class Batch:

    def __init__(
            self,
            inputs,
            outputs = None,
            lengths = None,
            misc = None):

        self.inputs = inputs
        self.outputs = outputs
        self.lengths = lengths
        self.misc = misc

    def __len__(self):
        return self.inputs.shape[1]

    def get_num_tokens(self):
        return sum(self.lengths)

    def cuda(self):
        self.inputs = self.inputs.cuda(non_blocking = True)

        if self.outputs is not None:
            self.outputs = self.outputs.cuda(non_blocking = True)

        return self

