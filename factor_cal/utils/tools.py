import os
import psutil

def show_info(start):
    pid = os.getpid()
    #模块名比较容易理解：获得当前进程的pid
    p = psutil.Process(pid)
	#根据pid找到进程，进而找到占用的内存值
    info = p.memory_full_info()
    memory = info.uss/1024/1024
    print(f'{start} 一共占用{memory:.2f}MB')
