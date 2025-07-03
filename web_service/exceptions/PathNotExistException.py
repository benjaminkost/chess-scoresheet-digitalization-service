class PathNotExistException(Exception):
    def __init__(self, path):
        self.path = path
    def __str__(self):
        return repr(self.path)
    def __repr__(self):
        return self.__str__()