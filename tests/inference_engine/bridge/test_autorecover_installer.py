from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "mac_bridge" / "install_autorecover_launchagent.sh"


def test_installer_uses_tcc_safe_support_directory():
    source = INSTALLER.read_text()
    assert 'SUPPORT_DIR="${HOME}/Library/Application Support/Kakeya"' in source
    assert 'install -m 755 "$RECOVER_SCRIPT" "$INSTALLED_RECOVER_SCRIPT"' in source
    assert "<string>${INSTALLED_RECOVER_SCRIPT}</string>" in source
    assert "<string>${HOME}/actions-runner</string>" in source


def test_installer_preserves_direct_runner_child():
    source = INSTALLER.read_text()
    assert "<key>AbandonProcessGroup</key>" in source
    assert "<true/>" in source
