from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "deploy" / "install_prefill_worker_launchd.sh"
HEAD_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.grpc-runtime-prefill.plist"


def test_worker_installer_emits_full_cache_compatibility_contract():
    source = INSTALLER.read_text()
    for flag in (
        "--sink",
        "--window",
        "--block-size-tokens",
        "--prefill-tps",
        "--network",
        "--priority",
        "--rtt-ms",
        "--max-concurrent-jobs",
        "--max-prompt-tokens",
    ):
        assert f"<string>{flag}</string>" in source
    assert 'PEER="${KAKEYA_WORKER_PEER:-}"' in source
    assert "<string>--peer</string>" in source


def test_head_runtime_discovers_and_uses_worker_cache_port():
    plist = HEAD_PLIST.read_text()
    assert (
        "<string>--peer</string><string>169.254.27.104:53051</string>"
        in plist
    )
    assert (
        "<string>--cache-peer</string><string>169.254.27.104:53051</string>"
        in plist
    )
    assert "<string>--primary-prefill-penalty-ms</string>" in plist
    assert (
        "<string>--cache-tenant-id</string><string>private-fleet</string>"
        in plist
    )
    assert "<string>--fleet-psk-file</string>" in plist
