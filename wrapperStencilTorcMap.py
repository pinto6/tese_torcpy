import os
os.system("mpirun -n 16 --mca btl_tcp_if_include bond0 stencilTorcMap.py")
