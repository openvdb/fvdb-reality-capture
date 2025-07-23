# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import random

from ._adjectives import ADJECTIVES
from ._nouns import NOUNS


def generate_name() -> str:
    """
    Generate a random human readable name by choosing a random adjective and noun,
    and formatting them into a readable string {adjective}_{noun}.

    Returns:
        str: A human readable name in the format "adjective_noun".
    """

    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adjective}_{noun}"


__all__ = ["generate_name"]
