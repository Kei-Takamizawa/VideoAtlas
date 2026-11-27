"""Browser GUI for VideoAtlas.

The web package serves pages on a loopback address only and starts analysis as
``python -m videoatlas.cli`` subprocesses. It never imports the analysis engine,
so a model failure cannot stop the GUI.
"""
