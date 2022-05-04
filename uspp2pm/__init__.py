import sys

def config_dependencies(is_kaggle):
    if is_kaggle:
        sys.path.append("/kaggle/input/uspp2pm/dependencies")
    else:
        sys.path.append("./dependencies")