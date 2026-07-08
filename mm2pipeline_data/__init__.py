"""mm2pipeline_data — the Mario Maker 2 format-conversion pipeline.

One package for every step that moves data through the project, anchored on the
canonical training tileset ``mm2_tileset_we.json`` (see ``mm2pipeline_data.tiles``).

Stages
------
    extract   HuggingFace mm2_level dataset   -> .bcd        (mm2pipeline_data.extract)
    toost     .bcd                            -> .json + .png (mm2pipeline_data.toost)
    ascii     .json  <->  ASCII grid          (mm2pipeline_data.ascii)
    dataset   ASCII  ->  tile-id dataset + split (mm2pipeline_data.dataset)
    swe       .json  ->  .swe (playable)      (mm2pipeline_data.swe)

``python -m mm2pipeline_data <command>`` (see ``mm2pipeline_data.__main__``) dispatches to
the stages; each stage module also exposes a ``main()`` so it runs directly as
``python -m mm2pipeline_data.<stage>``. The shared object metadata lives in
``mm2pipeline_data.tiles``; the binary .bcd codec lives in ``mm2pipeline_data.bcd``.
See mm2pipeline_data/README.md for full usage instructions.
"""

__version__ = "1.1.0"
