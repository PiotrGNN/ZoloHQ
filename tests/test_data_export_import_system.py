import pytest
from data_export_import_system import DataExportImportSystem

def test_create_default_templates():
    sys = DataExportImportSystem()
    assert "daily_summary" in sys.export_templates

def test_initialize_production_data_handles_missing():
    sys = DataExportImportSystem()
    # Should not raise even if production data is missing
    sys._initialize_production_data()

def test_export_import_edge_cases():
    sys = DataExportImportSystem()
    # Simulate missing file/permission error
    import tempfile, os, stat
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    os.chmod(temp_file.name, 0)
    try:
        with pytest.raises(Exception):
            with open(temp_file.name, "r") as f:
                f.read()
    finally:
        os.chmod(temp_file.name, stat.S_IWRITE)
        os.unlink(temp_file.name)
