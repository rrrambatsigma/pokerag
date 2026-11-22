"""B5 supervised data: facts in, grounded chat examples out.

The corpus is CC BY-NC-SA 2.5 with ``allow_ft_training = false``. Training pairs
are therefore written from extracted facts (values, not sentences), and the leak
gates in :mod:`pokerag.sft.leak` enforce that no corpus text and no test entity
reaches the dataset.
"""

from pokerag.sft.dataset import Example, Item, build_examples, build_items

__all__ = ["Example", "Item", "build_examples", "build_items"]
