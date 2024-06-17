import os
import psutil
import inspect

def show_memory(start):
    pid = os.getpid()
    # obtain the current process id
    p = psutil.Process(pid)
    # according to the pid, find the memory value occupied
    info = p.memory_full_info()
    memory = info.uss/1024/1024
    print(f'{start} 一共占用{memory:.2f}MB')



def get_func_info(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    
    args = []
    kwargs = []
    
    for param in parameters.values():
        if param.default == inspect.Parameter.empty:
            args.append((param.name, param.annotation))
        else:
            kwargs.append((param.name, param.annotation, param.default))
    
    return {
        'name': func.__name__,
        'return_type': signature.return_annotation,
        'args': args,
        'kwargs': kwargs
    }
