from importlib import import_module

def resolve_class(name: str, module: str):
    """
    module: 클래스가 저장되어 있는 위치 
    name: 클래스의 이름 
    """
    mod = import_module(module)
    return getattr(mod, name)
