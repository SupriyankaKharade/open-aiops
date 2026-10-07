import open_aiops


def test_package_metadata():
    """Verify package version and basic metadata exports (INT-1)."""
    assert hasattr(open_aiops, "__version__")
    assert open_aiops.__version__ == "0.1.0"
    assert hasattr(open_aiops, "__title__")
    assert open_aiops.__title__ == "open-aiops"
    assert hasattr(open_aiops, "__author__")
    assert open_aiops.__author__ == "OpenAIOps Contributors"
    assert hasattr(open_aiops, "__license__")
    assert open_aiops.__license__ == "Apache-2.0"
    assert hasattr(open_aiops, "__description__")
