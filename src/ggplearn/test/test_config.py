from ggplearn import config

def test_simple():
    bt = config.Breakthrough()

    assert bt.nb_roles == 2
    assert bt.nb_rows == 8
    assert bt.nb_cols == 8

    print bt.nb_input_layers == 4
