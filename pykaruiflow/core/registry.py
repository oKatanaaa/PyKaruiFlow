MUL_OP_CLASS = None

def register_mul_op(cls):
    global MUL_OP_CLASS
    MUL_OP_CLASS = cls

def get_mul_op():
    assert MUL_OP_CLASS is not None, 'No mul op has been registered.'
    return MUL_OP_CLASS()


ADD_OP_CLASS = None

def register_add_op(cls):
    global ADD_OP_CLASS
    ADD_OP_CLASS = cls

def get_add_op():
    assert ADD_OP_CLASS is not None, 'No add op has been registered.'
    return ADD_OP_CLASS()