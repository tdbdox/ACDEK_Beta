from dataclasses import make_dataclass

def generate_dataclass(name: str, traits: dict):
    fields = [(k, type(v)) for k, v in traits.items()]
    return make_dataclass(name, fields)