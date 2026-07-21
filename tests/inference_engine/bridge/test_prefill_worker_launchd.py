from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "deploy" / "install_prefill_worker_launchd.sh"
HEAD_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.grpc-runtime-prefill.plist"
PEER_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.prefill-network-peer.plist"
WORKER_PLIST = ROOT / "deploy" / "launchd" / "ai.kakeya.prefill-worker-peer.plist"
WATCHDOG_INSTALLER = ROOT / "deploy" / "install_decode_watchdog_launchd.sh"


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
        "--cache-min-gb",
        "--memory-reserve-gb",
        "--estimated-snapshot-bytes-per-token",
        "--prefill-compute-chunk-tokens",
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
    assert "<string>--bind</string><string>0.0.0.0:51051</string>" in plist
    assert (
        "<string>--prefill-worker-timeout-s</string><string>3600</string>"
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
    assert "<string>--cache-gb</string><string>8</string>" in worker
    assert "<string>--cache-min-gb</string><string>1</string>" in worker
    assert "<string>--memory-reserve-gb</string><string>0.5</string>" in worker
    assert (
        "<string>--estimated-snapshot-bytes-per-token</string>"
        "<string>400000</string>"
        in worker
    )
    assert (
        "<string>--prefill-compute-chunk-tokens</string>"
        "<string>256</string>"
        in worker
    )
    assert "<string>--adaptive-cache</string>" in worker
    assert "<string>--window</string><string>2048</string>" in worker
    assert "<string>--prefill-tps</string><string>1</string>" in worker
    assert "scripts/start_prefill_cache_node.py" in PEER_PLIST.read_text()


def test_worker_injects_explicit_retained_token_cap():
    source = (
        ROOT / "scripts" / "start_prefill_worker_node.py"
    ).read_text()
    assert "max_retained_tokens=args.sink + args.window" in source


def test_decode_watchdog_launchagent_is_external_and_restarts_primary():
    source = WATCHDOG_INSTALLER.read_text()
    assert "scripts/decode_watchdog.py" in source
    assert "<key>StartInterval</key>" in source
    assert 'STALL_SECONDS="${KAKEYA_DECODE_STALL_SECONDS:-120}"' in source
    assert "<string>--runtime-label</string>" in source
    assert 'launchctl kickstart -k "$DOMAIN/$WATCHDOG_LABEL"' in source
