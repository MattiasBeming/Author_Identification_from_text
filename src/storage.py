
class DS_obj:
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid

    def set_valid(self, valid):
        self.valid = valid

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def get_valid(self):
        if not self.valid:
            raise ValueError('Validation data not set')
        return self.valid


class Dataset:
    def __init__(self, name, datasets={}):
        self.name = name
        self.datasets = datasets

    def _get_ds(self, name):
        try:
            return self.datasets[name]
        except KeyError:
            raise KeyError(f'{name} does not exist in Dataset')

    def add_dataset(self, name, train, test, valid=None):
        self.datasets[name] = DS_obj(train, test, valid)
    
    def set_valid(self, name, valid):
        ds = self._get_ds(name)
        ds.set_valid(valid)

    def get_name(self):
        return self.name
    
    def get_datasets_dict(self):
        return self.datasets

    def get_train(self, name):
        return self._get_ds(name).get_train()
    
    def get_test(self, name):
        return self._get_ds(name).get_test()
    
    def get_valid(self, name):
        return self._get_ds(name).get_valid()
