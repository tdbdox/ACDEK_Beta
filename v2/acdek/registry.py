_semantic_class_registry = {}

def register_semantic_class(name: str, cls):
    _semantic_class_registry[name] = cls

def get_registered_classes():
    return _semantic_class_registry