from DataLoaders import base



class DIV2K(base.AbstractDateset):
    def __init__(self, args, data_type):

        self.args = args
        
        super(DIV2K, self).__init__(data_type)

