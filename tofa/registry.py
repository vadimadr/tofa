class Registry:
    def __init__(self, name):
        self._name = name
        self._registry_dict = dict()

    def register(self, name=None, obj=None):
        if obj is not None:
            if name is None:
                name = obj.__name__
            return self._register(obj, name)
        return self._decorate(name)

    def get(self, name):
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _register(self, obj, name):
        if name in self._registry_dict:
            raise KeyError("{} is already registered in {}".format(name, self._name))
        self._registry_dict[name] = obj

    def _decorate(self, name=None):
        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            self._register(obj, cls_name)
            return obj

        return wrap

    def _key_not_found(self, name):
        raise KeyError("{} is unknown type of {} ".format(name, self._name))

    @property
    def registry_dict(self):
        return self._registry_dict
