
def test_cffi_to_cmodule():
    from ggplib.db import lookup
    from ggplib.interface import ffi
    import ggpzero_interface
    wrap = ggpzero_interface.start(42,42)

    info = lookup.by_name("breakthrough")

    cffi_ptr = info.get_sm().c_statemachine
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    print wrap.test_sm(ptr_as_long)
