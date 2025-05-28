import math



class DoNothing:
    def forward(self, vit_embeds, *args, **kwargs):
        return vit_embeds

def nearest_square(num: int) -> int:
    """
    Find the nearest square number to the given integer.

    Parameters:
    ----------
    num : int
        The input integer.

    Returns:
    -------
    int
        The nearest square number.
    """
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")

    # Find the square root of the number and round it
    root = round(math.sqrt(num))
    # Return the square of the rounded root
    return root ** 2