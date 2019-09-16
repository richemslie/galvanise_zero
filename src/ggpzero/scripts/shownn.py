import sys

from ggpzero.nn.manager import get_manager

if __name__ == "__main__":
    def main(args):
        game = args[0]
        gen = args[1]

        man = get_manager()
        nn = man.load_network(game, gen)
        nn.summary()

    from ggpzero.util.main import main_wrap
    main_wrap(main)
