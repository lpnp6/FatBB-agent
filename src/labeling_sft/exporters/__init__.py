__all__ = ["GGUFExporter"]


def __getattr__(name: str):
    if name == "GGUFExporter":
        from labeling_sft.exporters.gguf import GGUFExporter
        return GGUFExporter
    raise AttributeError(name)
