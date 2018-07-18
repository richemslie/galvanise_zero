import os
from cffi import FFI
from ggplib import interface


def back(path, depth):
    for i in range(depth):
        path = os.path.dirname(path)
    return path


def process_line(line):
    # pre-process a line.  Skip any lines with comments.  Replace strings in remap.
    if "//" in line:
        return line

    remap = {
        "StateMachine*" : "void*",
        "PlayerBase*" : "void*",
        "boolean" : "int",
    }

    for k, v in remap.items():
        if k in line:
            line = line.replace(k, v)
            line = line.rstrip()

    return line


def get_lines(filename):
    # take subset of file (since it is c++, and want only the c portion
    emit = False
    for line in file(filename):
        if "CFFI START INCLUDE" in line:
            emit = True

        elif "CFFI END INCLUDE" in line:
            emit = False

        if emit:
            line = process_line(line)
            if line:
                yield line


def get_lib():
    # get the paths
    ggplib_path = os.path.join(os.environ["GGPLIB_PATH"], "src", "cpp")
    local_path = os.path.join(os.environ["GURGEH_PATH"], "src", "cpp")

    print ggplib_path, local_path

    # get ffi object, and lib object
    ffi = FFI()
    ffi.cdef("\n".join(get_lines(os.path.join(local_path, "interface.h"))))

    return ffi, ffi.verify('#include <interface.h>\n',
                           include_dirs=[local_path],
                           library_dirs=[ggplib_path, local_path],
                           libraries=["gurgehplayer_cpp"])


_, lib = get_lib()


def create_gurgeh_cpp_player(sm, our_role_index, *args):
    return interface.CppPlayerWrapper(lib.Player__createGurgehPlayer(sm.c_statemachine,
                                                                     our_role_index,
                                                                     *args))
