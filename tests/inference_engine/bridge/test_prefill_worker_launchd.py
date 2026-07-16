from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "deploy" / "install_prefill_worker_launchd.sh"
HEAD_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.grpc-runtime-prefill.plist"
PEER_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.prefill-network-peer.plist"
WORKER_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.prefill-worker-peer.plist"


def test_worker_installer_emits_full_cache_compatibility_contract():
    source = INSTALLER.read_text()
    for flag in (
        "--sink",
        "--window",
        "--block-size-tokens",
        "--cache-format-version",
        "--cache-compression",
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
    assert 'chmod 644 "$PLIST"' in source
    assert 'launchctl print "$DOMAIN/$LABEL"' in source
    assert "for attempt in 1 2 3" in source
    assert 'launchctl kickstart -k "$DOMAIN/$LABEL"' in source


def test_two_mac_deployment_uses_allens_as_prefill_only():
    plist = HEAD_PLIST.read_text()
    assert (
        "<string>--peer</string><string>169.254.27.104:53051</string>"
        in plist
    )
    assert (
        "<string>--cache-peer</string><string>169.254.27.104:53051</string>"
        in plist
    )
    assert (
        "<string>--prefill-policy</string><string>remote-required</string>"
        in plist
    )
    assert (
        "<string>--cache-tenant-id</string><string>private-fleet</string>"
        in plist
    )
    assert "<string>--fleet-psk-file</string>" in plist
    assert (
        "<string>--cache-compression</string>"
        "<string>kakeyalattice-d4</string>"
        in plist
    )
    worker = WORKER_PLIST.read_text()
    assert "scripts/start_prefill_worker_node.py" in worker
    assert "scripts/start_prefill_cache_node.py" not in worker
    assert "<string>--cache-gb</string><string>0.25</string>" in worker
    assert "<string>--window</string><string>2048</string>" in worker
    assert "<string>--prefill-tps</string><string>1</string>" in worker
    assert "scripts/start_prefill_cache_node.py" in PEER_PLIST.read_text()
