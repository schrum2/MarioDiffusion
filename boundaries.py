# 224 is the highest height of the original levels before the sky buffer is added
# Nothing of interest can appear above this line, which is why the height of interest
# is lowered.
TOP_LINE = 32 + 224 / 3
BOTTOM_LINE = 256 - (224 / 3)

LEFT_LINE = 256 / 3
RIGHT_LINE = 256 - (256 / 3)