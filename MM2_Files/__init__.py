"""Mario Maker 2 support modules (captioner, LLM captioner, renderer).

Making MM2_Files a package lets the rest of the repo import these modules as
``from MM2_Files.<module> import ...`` -- the same way util/ and mm2pipeline_data/
are imported to avoid the need to modify sys.path or use relative imports. This works regardless of where this module is invoked from.
"""
