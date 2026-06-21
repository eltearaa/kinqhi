"""Tests for hermes backup and import commands."""

import json
import os
import sqlite3
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hermes_tree(root: Path) -> None:
    """Create a realistic ~/.kinqhi directory structure for testing."""
    (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    (root / "memory_store.db").write_bytes(b"fake-sqlite")
    (root / "kinqhi_state.db").write_bytes(b"fake-state")

    # Sessions
    (root / "sessions").mkdir(exist_ok=True)
    (root / "sessions" / "abc123.json").write_text("{}")

    # Skills
    (root / "skills").mkdir(exist_ok=True)
    (root / "skills" / "my-skill").mkdir()
    (root / "skills" / "my-skill" / "SKILL.md").write_text("# My Skill\n")

    # Skins
    (root / "skins").mkdir(exist_ok=True)
    (root / "skins" / "cyber.yaml").write_text("name: cyber\n")

    # Cron
    (root / "cron").mkdir(exist_ok=True)
    (root / "cron" / "jobs.json").write_text("[]")

    # Memories
    (root / "memories").mkdir(exist_ok=True)
    (root / "memories" / "notes.json").write_text("{}")

    # Profiles
    (root / "profiles").mkdir(exist_ok=True)
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "coder" / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (root / "profiles" / "coder" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-123\n")

    # kinqhi repo (should be EXCLUDED)
    (root / "kinqhi").mkdir(exist_ok=True)
    (root / "kinqhi" / "run_agent.py").write_text("# big file\n")
    (root / "kinqhi" / ".git").mkdir()
    (root / "kinqhi" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    # __pycache__ (should be EXCLUDED)
    (root / "plugins").mkdir(exist_ok=True)
    (root / "plugins" / "__pycache__").mkdir()
    (root / "plugins" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")

    # PID files (should be EXCLUDED)
    (root / "gateway.pid").write_text("12345")

    # Logs (should be included)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "agent.log").write_text("log line\n")


def _symlink_file_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


# ---------------------------------------------------------------------------
# _should_exclude tests
# ---------------------------------------------------------------------------

class TestShouldExclude:
    def test_excludes_hermes_agent(self):
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("kinqhi/run_agent.py"))
        assert _should_exclude(Path("kinqhi/.git/HEAD"))

    def test_excludes_pycache(self):
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("plugins/__pycache__/mod.cpython-312.pyc"))

    def test_excludes_pyc_files(self):
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("some/module.pyc"))

    def test_excludes_pid_files(self):
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("gateway.pid"))
        assert _should_exclude(Path("cron.pid"))

    def test_excludes_checkpoints(self):
        """checkpoints/ is session-local trajectory cache — hash-keyed,
        regenerated per-session, won't port to another machine anyway."""
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("checkpoints/abc123/trajectory.json"))
        assert _should_exclude(Path("checkpoints/deadbeef/step_0001.json"))

    def test_excludes_backups_dir(self):
        """backups/ is excluded so pre-update backups don't nest exponentially."""
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("backups/pre-update-2026-04-27-063400.zip"))

    def test_excludes_sqlite_sidecars(self):
        """SQLite WAL/SHM/journal sidecars must not ship alongside the
        safe-copied .db — pairing a fresh snapshot with stale sidecar state
        produces a torn restore."""
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path("state.db-wal"))
        assert _should_exclude(Path("state.db-shm"))
        assert _should_exclude(Path("state.db-journal"))
        assert _should_exclude(Path("memory_store.db-wal"))
        # The .db itself is still included (and safe-copied separately)
        assert not _should_exclude(Path("state.db"))

    def test_includes_config(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("config.yaml"))

    def test_includes_env(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path(".env"))

    def test_includes_skills(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/my-skill/SKILL.md"))

    def test_includes_profiles(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("profiles/coder/config.yaml"))

    def test_includes_sessions(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("sessions/abc.json"))

    def test_includes_logs(self):
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("logs/agent.log"))

    def test_includes_nested_hermes_agent_in_skills(self):
        """skills/autonomous-ai-agents/kinqhi/ must NOT be excluded —
        only the root-level kinqhi/ repo is skipped."""
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/autonomous-ai-agents/kinqhi/SKILL.md"))
        assert not _should_exclude(Path("skills/autonomous-ai-agents/kinqhi/sub/item.txt"))

    @pytest.mark.parametrize(
        "rel",
        [
            "plugins/my-plugin/.venv/lib/python3.12/site-packages/x/__init__.py",
            "plugins/my-plugin/venv/bin/python",
            "mcp/server/site-packages/pkg/mod.py",
            ".cache/uv/wheels/abc.whl",
            "plugins/p/.cache/pip/http/deadbeef",
            ".tox/py312/log.txt",
            ".nox/tests/bin/pytest",
            "plugins/p/.pytest_cache/v/cache/lastfailed",
            ".mypy_cache/3.12/agent.meta.json",
            ".ruff_cache/0.4.0/abc",
        ],
    )
    def test_excludes_regeneratable_dependency_and_cache_dirs(self, rel):
        """Python dep trees and tool caches under KINQHI_HOME must be skipped —
        these are what balloon a backup to hundreds of thousands of files."""
        from kinqhi_cli.backup import _should_exclude
        assert _should_exclude(Path(rel))

    def test_does_not_exclude_curator_archive(self):
        """skills/.archive/ holds restorable archived skills and MUST survive
        a backup — it is intentionally NOT in the exclusion set."""
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/.archive/old-skill/SKILL.md"))

    def test_does_not_exclude_legit_files_resembling_cache_names(self):
        """Only directory-component matches are excluded; a normal file is kept."""
        from kinqhi_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/my-skill/venv-notes.md"))
        assert not _should_exclude(Path("memories/cache.json"))

# ---------------------------------------------------------------------------
# Backup tests
# ---------------------------------------------------------------------------

class TestBackup:
    def test_creates_zip(self, tmp_path, monkeypatch):
        """Backup creates a valid zip containing expected files."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        # get_default_hermes_root needs this
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            # Config should be present
            assert "config.yaml" in names
            assert ".env" in names
            # Skills
            assert "skills/my-skill/SKILL.md" in names
            # Profiles
            assert "profiles/coder/config.yaml" in names
            assert "profiles/coder/.env" in names
            # Sessions
            assert "sessions/abc123.json" in names
            # Logs
            assert "logs/agent.log" in names
            # Skins
            assert "skins/cyber.yaml" in names

    def test_db_snapshots_staged_beside_output_zip(self, tmp_path, monkeypatch):
        """SQLite staging temp files must be created on the output zip's
        filesystem (dir=out_path.parent), NOT the system /tmp default — a
        small tmpfs there silently drops large DBs from the backup (#35376)."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_dir = tmp_path / "external-drive"
        out_dir.mkdir()
        out_zip = out_dir / "backup.zip"
        args = Namespace(output=str(out_zip))

        import kinqhi_cli.backup as backup_mod
        staged_dirs = []
        real_ntf = backup_mod.tempfile.NamedTemporaryFile

        def _spy(*a, **kw):
            staged_dirs.append(kw.get("dir"))
            return real_ntf(*a, **kw)

        monkeypatch.setattr(backup_mod.tempfile, "NamedTemporaryFile", _spy)
        backup_mod.run_backup(args)

        # At least one .db was staged, and every staging call targeted the
        # output zip's directory rather than the system temp default.
        assert staged_dirs, "no SQLite snapshot was staged"
        assert all(d == str(out_dir) for d in staged_dirs), staged_dirs

    def test_pre_update_db_snapshots_staged_beside_output_zip(self, tmp_path, monkeypatch):
        """The pre-update/pre-migration zip path (_write_full_zip_backup) must
        also stage SQLite snapshots beside its output zip, not in /tmp."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = kinqhi_home / "backups" / "pre-update-test.zip"
        out_zip.parent.mkdir(parents=True, exist_ok=True)

        import kinqhi_cli.backup as backup_mod
        staged_dirs = []
        real_ntf = backup_mod.tempfile.NamedTemporaryFile

        def _spy(*a, **kw):
            staged_dirs.append(kw.get("dir"))
            return real_ntf(*a, **kw)

        monkeypatch.setattr(backup_mod.tempfile, "NamedTemporaryFile", _spy)
        result = backup_mod._write_full_zip_backup(out_zip, kinqhi_home)

        assert result is not None
        assert staged_dirs, "no SQLite snapshot was staged"
        assert all(d == str(out_zip.parent) for d in staged_dirs), staged_dirs

    def test_excludes_hermes_agent(self, tmp_path, monkeypatch):
        """Backup does NOT include kinqhi/ directory."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            agent_files = [n for n in names if "kinqhi" in n]
            assert agent_files == [], f"kinqhi files leaked into backup: {agent_files}"

    def test_excludes_dependency_and_cache_trees(self, tmp_path, monkeypatch):
        """A plugin venv / site-packages / pip cache under KINQHI_HOME must be
        pruned by the walk, while real data (skills, config) is preserved.
        This is the regression guard for the ballooning-backup bug."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        # Simulate the heavy regeneratable trees that ballooned the backup.
        venv_pkg = kinqhi_home / "plugins" / "heavy" / ".venv" / "lib" / "site-packages" / "dep"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "__init__.py").write_text("# dep\n")
        pip_cache = kinqhi_home / ".cache" / "uv" / "wheels"
        pip_cache.mkdir(parents=True)
        (pip_cache / "abc.whl").write_bytes(b"\x00")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        from kinqhi_cli.backup import run_backup
        run_backup(Namespace(output=str(out_zip)))

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
        leaked = [n for n in names if ".venv" in n or "site-packages" in n or ".cache" in n]
        assert leaked == [], f"regeneratable trees leaked into backup: {leaked}"
        # Real data still present.
        assert "skills/my-skill/SKILL.md" in names
        assert "config.yaml" in names

    def test_includes_nested_hermes_agent_in_skills(self, tmp_path, monkeypatch):
        """Backup includes skills/.../kinqhi/ but NOT root kinqhi/."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        # Add a nested kinqhi directory inside skills (like the real layout)
        nested = kinqhi_home / "skills" / "autonomous-ai-agents" / "kinqhi"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("# Kinqhi Skill\n")
        (nested / "sub").mkdir()
        (nested / "sub" / "item.txt").write_text("nested content\n")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            # Root kinqhi must be excluded
            root_agent = [n for n in names if n.startswith("kinqhi/")]
            assert root_agent == [], f"root kinqhi leaked: {root_agent}"
            # Nested skill kinqhi must be included
            assert "skills/autonomous-ai-agents/kinqhi/SKILL.md" in names
            assert "skills/autonomous-ai-agents/kinqhi/sub/item.txt" in names

    def test_excludes_pycache(self, tmp_path, monkeypatch):
        """Backup does NOT include __pycache__ dirs."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            pycache_files = [n for n in names if "__pycache__" in n]
            assert pycache_files == []

    def test_excludes_pid_files(self, tmp_path, monkeypatch):
        """Backup does NOT include PID files."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            pid_files = [n for n in names if n.endswith(".pid")]
            assert pid_files == []

    def test_default_output_path(self, tmp_path, monkeypatch):
        """When no output path given, zip goes to ~/hermes-backup-*.zip."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        args = Namespace(output=None)

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        # Should exist in home dir
        zips = list(tmp_path.glob("hermes-backup-*.zip"))
        assert len(zips) == 1

    def test_skips_symlinked_files(self, tmp_path, monkeypatch):
        """Backup must not dereference symlinks and leak files outside KINQHI_HOME."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        _make_hermes_tree(kinqhi_home)
        outside = tmp_path / "outside-secret.txt"
        outside.write_text("outside secret\n")
        _symlink_file_or_skip(kinqhi_home / "skills" / "outside-link.txt", outside)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert "skills/outside-link.txt" not in names
            assert all(zf.read(name) != b"outside secret\n" for name in names)


# ---------------------------------------------------------------------------
# _validate_backup_zip tests
# ---------------------------------------------------------------------------

class TestValidateBackupZip:
    def _make_zip(self, zip_path: Path, filenames: list[str]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in filenames:
                zf.writestr(name, "dummy")

    def test_state_db_passes(self, tmp_path):
        """A zip containing state.db is accepted as a valid Kinqhi backup."""
        from kinqhi_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "backup.zip"
        self._make_zip(zip_path, ["state.db", "sessions/abc.json"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok, reason

    def test_old_wrong_db_name_fails(self, tmp_path):
        """A zip with only kinqhi_state.db (old wrong name) is rejected."""
        from kinqhi_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "old.zip"
        self._make_zip(zip_path, ["kinqhi_state.db", "memory_store.db"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert not ok

    def test_config_yaml_passes(self, tmp_path):
        """A zip containing config.yaml is accepted (existing behaviour preserved)."""
        from kinqhi_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "backup.zip"
        self._make_zip(zip_path, ["config.yaml", "skills/x/SKILL.md"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok, reason


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        """Create a test zip with given files."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                if isinstance(content, bytes):
                    zf.writestr(name, content)
                else:
                    zf.writestr(name, content)

    def test_restores_files(self, tmp_path, monkeypatch):
        """Import extracts files into hermes home."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model:\n  provider: openrouter\n",
            ".env": "OPENROUTER_API_KEY=sk-test\n",
            "skills/my-skill/SKILL.md": "# My Skill\n",
            "profiles/coder/config.yaml": "model:\n  provider: anthropic\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        assert (kinqhi_home / "config.yaml").read_text() == "model:\n  provider: openrouter\n"
        assert (kinqhi_home / ".env").read_text() == "OPENROUTER_API_KEY=sk-test\n"
        assert (kinqhi_home / "skills" / "my-skill" / "SKILL.md").read_text() == "# My Skill\n"
        assert (kinqhi_home / "profiles" / "coder" / "config.yaml").exists()

    def test_strips_hermes_prefix(self, tmp_path, monkeypatch):
        """Import strips .hermes/ prefix if all entries share it."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            ".hermes/config.yaml": "model: test\n",
            ".hermes/skills/a/SKILL.md": "# A\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        assert (kinqhi_home / "config.yaml").read_text() == "model: test\n"
        assert (kinqhi_home / "skills" / "a" / "SKILL.md").read_text() == "# A\n"

    def test_rejects_empty_zip(self, tmp_path, monkeypatch):
        """Import rejects an empty zip."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass  # empty

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_rejects_non_hermes_zip(self, tmp_path, monkeypatch):
        """Import rejects a zip that doesn't look like a hermes backup."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "random.zip"
        self._make_backup_zip(zip_path, {
            "some/random/file.txt": "hello",
            "another/thing.json": "{}",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_blocks_path_traversal(self, tmp_path, monkeypatch):
        """Import blocks zip entries with path traversal."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "evil.zip"
        # Include a marker file so validation passes
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "../../etc/passwd": "root:x:0:0\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # config.yaml should be restored
        assert (kinqhi_home / "config.yaml").exists()
        # traversal file should NOT exist outside hermes home
        assert not (tmp_path / "etc" / "passwd").exists()

    def test_preserves_live_gateway_state(self, tmp_path, monkeypatch):
        """Import must not overwrite the target's gateway_state.json.

        The backup carries the *source* machine's gateway run/desired state.
        Restoring it onto a hosted container drives the boot reconciler off
        stale/foreign state and leaves the gateway stuck "starting",
        disconnecting it from the Nous portal (NS-508). The live file wins.
        """
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # The target (e.g. hosted container) already has its own live state.
        live_state = '{"gateway_state": "running", "desired_state": "running"}'
        (kinqhi_home / "gateway_state.json").write_text(live_state)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            # A backup from a laptop where the gateway was stopped.
            "gateway_state.json": '{"gateway_state": "stopped", "desired_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # config.yaml is restored normally...
        assert (kinqhi_home / "config.yaml").read_text() == "model: test\n"
        # ...but the live gateway_state.json is untouched.
        assert (kinqhi_home / "gateway_state.json").read_text() == live_state

    def test_does_not_seed_gateway_state_when_absent(self, tmp_path, monkeypatch):
        """A backup's gateway_state.json is dropped, not written, when the
        target has none — a foreign state must never seed the reconciler."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "gateway_state.json": '{"gateway_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        assert (kinqhi_home / "config.yaml").exists()
        assert not (kinqhi_home / "gateway_state.json").exists()

    def test_preserves_per_profile_gateway_state(self, tmp_path, monkeypatch):
        """The skip is matched by basename, so a named profile's
        gateway_state.json (profiles/<name>/gateway_state.json) is preserved
        the same way the root profile's is."""
        kinqhi_home = tmp_path / ".hermes"
        (kinqhi_home / "profiles" / "coder").mkdir(parents=True)
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_state = '{"gateway_state": "running"}'
        (kinqhi_home / "profiles" / "coder" / "gateway_state.json").write_text(live_state)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/coder/config.yaml": "model: anthropic\n",
            "profiles/coder/gateway_state.json": '{"gateway_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # Profile config is restored, but its live gateway state is preserved.
        assert (kinqhi_home / "profiles" / "coder" / "config.yaml").read_text() == "model: anthropic\n"
        assert (
            kinqhi_home / "profiles" / "coder" / "gateway_state.json"
        ).read_text() == live_state

    def test_preserves_runtime_pid_and_process_files(self, tmp_path, monkeypatch):
        """gateway.pid / cron.pid / gateway.lock / processes.json from a backup
        reference the source machine's process namespace and must never be
        written over the target's."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Live runtime files belonging to the target's own processes.
        (kinqhi_home / "gateway.pid").write_text("4242")
        (kinqhi_home / "processes.json").write_text('{"live": true}')

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "gateway.pid": "9999",
            "cron.pid": "8888",
            "gateway.lock": "7777",
            "processes.json": '{"stale": true}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # Live runtime files are untouched; the backup's foreign ones never land.
        assert (kinqhi_home / "gateway.pid").read_text() == "4242"
        assert (kinqhi_home / "processes.json").read_text() == '{"live": true}'
        # cron.pid / gateway.lock had no live copy and were not seeded.
        assert not (kinqhi_home / "cron.pid").exists()
        assert not (kinqhi_home / "gateway.lock").exists()

    def test_confirmation_prompt_abort(self, tmp_path, monkeypatch):
        """Import aborts when user says no to confirmation."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        # Pre-existing config triggers the confirmation
        (kinqhi_home / "config.yaml").write_text("existing: true\n")
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: restored\n",
        })

        args = Namespace(zipfile=str(zip_path), force=False)

        from kinqhi_cli.backup import run_import
        with patch("builtins.input", return_value="n"):
            run_import(args)

        # Original config should be unchanged
        assert (kinqhi_home / "config.yaml").read_text() == "existing: true\n"

    def test_force_skips_confirmation(self, tmp_path, monkeypatch):
        """Import with --force skips confirmation and overwrites."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("existing: true\n")
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: restored\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        assert (kinqhi_home / "config.yaml").read_text() == "model: restored\n"

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        """Import exits with error for nonexistent file."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))

        args = Namespace(zipfile=str(tmp_path / "nonexistent.zip"), force=True)

        from kinqhi_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
    def test_restores_secret_files_with_0600_perms(self, tmp_path, monkeypatch):
        """Secret files must end up at 0600 after restore (zipfile drops mode bits)."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: openrouter\n",
            ".env": "OPENROUTER_API_KEY=sk-secret\n",
            "auth.json": '{"providers": {"nous": "token"}}',
            "state.db": b"SQLite format 3\x00",
            "profiles/coder/.env": "ANTHROPIC_API_KEY=sk-ant-secret\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        for rel in (".env", "auth.json", "state.db", "profiles/coder/.env"):
            mode = (kinqhi_home / rel).stat().st_mode & 0o777
            assert mode == 0o600, f"{rel} restored with mode {oct(mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_backup_then_import(self, tmp_path, monkeypatch):
        """Full round-trip: backup -> import to a new location -> verify."""
        # Source
        src_home = tmp_path / "source" / ".hermes"
        src_home.mkdir(parents=True)
        _make_hermes_tree(src_home)

        monkeypatch.setenv("KINQHI_HOME", str(src_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "source")

        # Backup
        out_zip = tmp_path / "roundtrip.zip"
        from kinqhi_cli.backup import run_backup, run_import

        run_backup(Namespace(output=str(out_zip)))
        assert out_zip.exists()

        # Import into a different location
        dst_home = tmp_path / "dest" / ".hermes"
        dst_home.mkdir(parents=True)
        monkeypatch.setenv("KINQHI_HOME", str(dst_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "dest")

        run_import(Namespace(zipfile=str(out_zip), force=True))

        # Verify key files
        assert (dst_home / "config.yaml").read_text() == "model:\n  provider: openrouter\n"
        assert (dst_home / ".env").read_text() == "OPENROUTER_API_KEY=sk-test-123\n"
        assert (dst_home / "skills" / "my-skill" / "SKILL.md").exists()
        assert (dst_home / "profiles" / "coder" / "config.yaml").exists()
        assert (dst_home / "sessions" / "abc123.json").exists()
        assert (dst_home / "logs" / "agent.log").exists()

        # kinqhi should NOT be present
        assert not (dst_home / "kinqhi").exists()
        # __pycache__ should NOT be present
        assert not (dst_home / "plugins" / "__pycache__").exists()
        # PID files should NOT be present
        assert not (dst_home / "gateway.pid").exists()


# ---------------------------------------------------------------------------
# Validate / detect-prefix unit tests
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_bytes(self):
        from kinqhi_cli.backup import _format_size
        assert _format_size(512) == "512 B"

    def test_kilobytes(self):
        from kinqhi_cli.backup import _format_size
        assert "KB" in _format_size(2048)

    def test_megabytes(self):
        from kinqhi_cli.backup import _format_size
        assert "MB" in _format_size(5 * 1024 * 1024)

    def test_gigabytes(self):
        from kinqhi_cli.backup import _format_size
        assert "GB" in _format_size(3 * 1024 ** 3)

    def test_terabytes(self):
        from kinqhi_cli.backup import _format_size
        assert "TB" in _format_size(2 * 1024 ** 4)


class TestValidation:
    def test_validate_with_config(self):
        """Zip with config.yaml passes validation."""
        import io
        from kinqhi_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.yaml", "test")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok

    def test_validate_with_env(self):
        """Zip with .env passes validation."""
        import io
        from kinqhi_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(".env", "KEY=val")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok

    def test_validate_rejects_random(self):
        """Zip without hermes markers fails validation."""
        import io
        from kinqhi_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("random/file.txt", "hello")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert not ok

    def test_detect_prefix_hermes(self):
        """Detects .hermes/ prefix wrapping all entries."""
        import io
        from kinqhi_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(".hermes/config.yaml", "test")
            zf.writestr(".hermes/skills/a/SKILL.md", "skill")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ".hermes/"

    def test_detect_prefix_none(self):
        """No prefix when entries are at root."""
        import io
        from kinqhi_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.yaml", "test")
            zf.writestr("skills/a/SKILL.md", "skill")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ""

    def test_detect_prefix_only_dirs(self):
        """Prefix detection returns empty for zip with only directory entries."""
        import io
        from kinqhi_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Only directory entries (trailing slash)
            zf.writestr(".hermes/", "")
            zf.writestr(".hermes/skills/", "")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ""


# ---------------------------------------------------------------------------
# Edge case tests for uncovered paths
# ---------------------------------------------------------------------------

class TestBackupEdgeCases:
    def test_nonexistent_kinqhi_home(self, tmp_path, monkeypatch):
        """Backup exits when hermes home doesn't exist."""
        fake_home = tmp_path / "nonexistent" / ".hermes"
        monkeypatch.setenv("KINQHI_HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "nonexistent")

        args = Namespace(output=str(tmp_path / "out.zip"))

        from kinqhi_cli.backup import run_backup
        with pytest.raises(SystemExit):
            run_backup(args)

    def test_output_is_directory(self, tmp_path, monkeypatch):
        """When output path is a directory, zip is created inside it."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_dir = tmp_path / "backups"
        out_dir.mkdir()

        args = Namespace(output=str(out_dir))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        zips = list(out_dir.glob("hermes-backup-*.zip"))
        assert len(zips) == 1

    def test_output_without_zip_suffix(self, tmp_path, monkeypatch):
        """Output path without .zip gets suffix appended."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_path = tmp_path / "mybackup.tar"
        args = Namespace(output=str(out_path))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        # Should have .tar.zip suffix
        assert (tmp_path / "mybackup.tar.zip").exists()

    def test_empty_kinqhi_home(self, tmp_path, monkeypatch):
        """Backup handles empty hermes home (no files to back up)."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        # Only excluded dirs, no actual files
        (kinqhi_home / "__pycache__").mkdir()
        (kinqhi_home / "__pycache__" / "foo.pyc").write_bytes(b"\x00")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        args = Namespace(output=str(tmp_path / "out.zip"))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        # No zip should be created
        assert not (tmp_path / "out.zip").exists()

    def test_permission_error_during_backup(self, tmp_path, monkeypatch):
        """Backup handles permission errors gracefully."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        # Create an unreadable file
        bad_file = kinqhi_home / "secret.db"
        bad_file.write_text("data")
        bad_file.chmod(0o000)

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "out.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        try:
            run_backup(args)
        finally:
            # Restore permissions for cleanup
            bad_file.chmod(0o644)

        # Zip should still be created with the readable files
        assert out_zip.exists()

    def test_pre1980_timestamp_skipped(self, tmp_path, monkeypatch):
        """Backup skips files with pre-1980 timestamps (ZIP limitation)."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        # Create a file with epoch timestamp (1970-01-01)
        old_file = kinqhi_home / "ancient.txt"
        old_file.write_text("old data")
        os.utime(old_file, (0, 0))

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "out.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        # Zip should still be created with the valid files
        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert "config.yaml" in names
            # The pre-1980 file should be skipped, not crash the backup
            assert "ancient.txt" not in names

    def test_skips_output_zip_inside_hermes(self, tmp_path, monkeypatch):
        """Backup skips its own output zip if it's inside hermes root."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Output inside hermes home
        out_zip = kinqhi_home / "backup.zip"
        args = Namespace(output=str(out_zip))

        from kinqhi_cli.backup import run_backup
        run_backup(args)

        # The zip should exist but not contain itself
        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            assert "backup.zip" not in zf.namelist()


class TestImportEdgeCases:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def test_not_a_zip(self, tmp_path, monkeypatch):
        """Import rejects a non-zip file."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))

        not_zip = tmp_path / "fake.zip"
        not_zip.write_text("this is not a zip")

        args = Namespace(zipfile=str(not_zip), force=True)

        from kinqhi_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_eof_during_confirmation(self, tmp_path, monkeypatch):
        """Import handles EOFError during confirmation prompt."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / "config.yaml").write_text("existing\n")
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "new\n"})

        args = Namespace(zipfile=str(zip_path), force=False)

        from kinqhi_cli.backup import run_import
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                run_import(args)

    def test_keyboard_interrupt_during_confirmation(self, tmp_path, monkeypatch):
        """Import handles KeyboardInterrupt during confirmation prompt."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        (kinqhi_home / ".env").write_text("KEY=val\n")
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "new\n"})

        args = Namespace(zipfile=str(zip_path), force=False)

        from kinqhi_cli.backup import run_import
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit):
                run_import(args)

    def test_permission_error_during_import(self, tmp_path, monkeypatch):
        """Import handles permission errors during extraction."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create a read-only directory so extraction fails
        locked_dir = kinqhi_home / "locked"
        locked_dir.mkdir()
        locked_dir.chmod(0o555)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "locked/secret.txt": "data",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        try:
            run_import(args)
        finally:
            locked_dir.chmod(0o755)

        # config.yaml should still be restored despite the error
        assert (kinqhi_home / "config.yaml").exists()

    def test_progress_with_many_files(self, tmp_path, monkeypatch):
        """Import shows progress with 500+ files."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "big.zip"
        files = {"config.yaml": "model: test\n"}
        for i in range(600):
            files[f"sessions/s{i:04d}.json"] = "{}"

        self._make_backup_zip(zip_path, files)

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        assert (kinqhi_home / "config.yaml").exists()
        assert (kinqhi_home / "sessions" / "s0599.json").exists()


# ---------------------------------------------------------------------------
# Profile restoration tests
# ---------------------------------------------------------------------------

class TestProfileRestoration:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def test_import_creates_profile_wrappers(self, tmp_path, monkeypatch):
        """Import auto-creates wrapper scripts for restored profiles."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock the wrapper dir to be inside tmp_path
        wrapper_dir = tmp_path / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model:\n  provider: openrouter\n",
            "profiles/coder/config.yaml": "model:\n  provider: anthropic\n",
            "profiles/coder/.env": "ANTHROPIC_API_KEY=sk-test\n",
            "profiles/researcher/config.yaml": "model:\n  provider: deepseek\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # Profile directories should exist
        assert (kinqhi_home / "profiles" / "coder" / "config.yaml").exists()
        assert (kinqhi_home / "profiles" / "researcher" / "config.yaml").exists()

        # Wrapper scripts should be created
        assert (wrapper_dir / "coder").exists()
        assert (wrapper_dir / "researcher").exists()

        # Wrappers should contain the right content
        coder_wrapper = (wrapper_dir / "coder").read_text()
        assert "hermes -p coder" in coder_wrapper

    def test_import_skips_profile_dirs_without_config(self, tmp_path, monkeypatch):
        """Import doesn't create wrappers for profile dirs without config."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        wrapper_dir = tmp_path / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/valid/config.yaml": "model: test\n",
            "profiles/empty/readme.txt": "nothing here\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from kinqhi_cli.backup import run_import
        run_import(args)

        # Only valid profile should get a wrapper
        assert (wrapper_dir / "valid").exists()
        assert not (wrapper_dir / "empty").exists()

    def test_import_without_profiles_module(self, tmp_path, monkeypatch):
        """Import gracefully handles missing profiles module (fresh install)."""
        kinqhi_home = tmp_path / ".hermes"
        kinqhi_home.mkdir()
        monkeypatch.setenv("KINQHI_HOME", str(kinqhi_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/coder/config.yaml": "model: test\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        # Simulate profiles module not being available
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *a, **kw):
            if name == "kinqhi_cli.profiles":
                raise ImportError("no profiles module")
            return original_import(name, *a, **kw)

        from kinqhi_cli.backup import run_import
        with patch("builtins.__import__", side_effect=fake_import):
            run_import(args)

        # Files should still be restored even if wrappers can't be created
        assert (kinqhi_home / "profiles" / "coder" / "config.yaml").exists()


# ---------------------------------------------------------------------------
# SQLite safe copy tests
# ---------------------------------------------------------------------------

class TestSafeCopyDb:
    def test_copies_valid_database(self, tmp_path):
        from kinqhi_cli.backup import _safe_copy_db
        src = tmp_path / "test.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
        conn.close()

        result = _safe_copy_db(src, dst)
        assert result is True

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [(42,)]

    def test_copies_wal_mode_database(self, tmp_path):
        from kinqhi_cli.backup import _safe_copy_db
        src = tmp_path / "wal.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES ('wal-test')")
        conn.commit()
        conn.close()

        result = _safe_copy_db(src, dst)
        assert result is True

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [("wal-test",)]


# ---------------------------------------------------------------------------
# Quick state snapshot tests
# ---------------------------------------------------------------------------

class TestQuickSnapshot:
    @pytest.fixture
    def kinqhi_home(self, tmp_path):
        """Create a fake KINQHI_HOME with critical state files."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text("model:\n  provider: openrouter\n")
        (home / ".env").write_text("OPENROUTER_API_KEY=test-key-123\n")
        (home / "auth.json").write_text('{"providers": {}}\n')
        (home / "channel_aliases.json").write_text(
            '{"whatsapp": {"120363408391911677@g.us": "general"}}\n'
        )
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text('{"jobs": []}\n')

        # Real SQLite database
        db_path = home / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('s1', 'hello world')")
        conn.commit()
        conn.close()
        return home

    def test_creates_snapshot(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        assert snap_id is not None
        snap_dir = kinqhi_home / "state-snapshots" / snap_id
        assert snap_dir.is_dir()
        assert (snap_dir / "manifest.json").exists()

    def test_label_in_id(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(label="before-upgrade", kinqhi_home=kinqhi_home)
        assert "before-upgrade" in snap_id

    def test_state_db_safely_copied(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        db_copy = kinqhi_home / "state-snapshots" / snap_id / "state.db"
        assert db_copy.exists()

        conn = sqlite3.connect(str(db_copy))
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("s1", "hello world")

    def test_copies_nested_files(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        assert (kinqhi_home / "state-snapshots" / snap_id / "cron" / "jobs.json").exists()

    def test_copies_channel_aliases(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        copied = kinqhi_home / "state-snapshots" / snap_id / "channel_aliases.json"
        assert copied.exists()
        assert "120363408391911677@g.us" in copied.read_text()

    def test_missing_files_skipped(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        with open(kinqhi_home / "state-snapshots" / snap_id / "manifest.json") as f:
            meta = json.load(f)
        # gateway_state.json etc. don't exist in fixture
        assert "gateway_state.json" not in meta["files"]

    def test_empty_home_returns_none(self, tmp_path):
        from kinqhi_cli.backup import create_quick_snapshot
        empty = tmp_path / "empty"
        empty.mkdir()
        assert create_quick_snapshot(kinqhi_home=empty) is None

    def test_list_snapshots(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, list_quick_snapshots
        id1 = create_quick_snapshot(label="first", kinqhi_home=kinqhi_home)
        id2 = create_quick_snapshot(label="second", kinqhi_home=kinqhi_home)

        snaps = list_quick_snapshots(kinqhi_home=kinqhi_home)
        assert len(snaps) == 2
        assert snaps[0]["id"] == id2  # most recent first
        assert snaps[1]["id"] == id1

    def test_list_limit(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, list_quick_snapshots
        for i in range(5):
            create_quick_snapshot(label=f"s{i}", kinqhi_home=kinqhi_home)
        snaps = list_quick_snapshots(limit=3, kinqhi_home=kinqhi_home)
        assert len(snaps) == 3

    def test_restore_config(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)

        (kinqhi_home / "config.yaml").write_text("model:\n  provider: anthropic\n")
        assert "anthropic" in (kinqhi_home / "config.yaml").read_text()

        result = restore_quick_snapshot(snap_id, kinqhi_home=kinqhi_home)
        assert result is True
        assert "openrouter" in (kinqhi_home / "config.yaml").read_text()

    def test_restore_state_db(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)

        conn = sqlite3.connect(str(kinqhi_home / "state.db"))
        conn.execute("INSERT INTO sessions VALUES ('s2', 'new')")
        conn.commit()
        conn.close()

        restore_quick_snapshot(snap_id, kinqhi_home=kinqhi_home)

        conn = sqlite3.connect(str(kinqhi_home / "state.db"))
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_restore_nonexistent(self, kinqhi_home):
        from kinqhi_cli.backup import restore_quick_snapshot
        assert restore_quick_snapshot("nonexistent", kinqhi_home=kinqhi_home) is False

    def test_auto_prune(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, list_quick_snapshots, _QUICK_DEFAULT_KEEP
        for i in range(_QUICK_DEFAULT_KEEP + 5):
            create_quick_snapshot(label=f"snap-{i:03d}", kinqhi_home=kinqhi_home)
        snaps = list_quick_snapshots(limit=100, kinqhi_home=kinqhi_home)
        assert len(snaps) <= _QUICK_DEFAULT_KEEP

    def test_manual_prune(self, kinqhi_home):
        from kinqhi_cli.backup import create_quick_snapshot, prune_quick_snapshots, list_quick_snapshots
        for i in range(10):
            create_quick_snapshot(label=f"s{i}", kinqhi_home=kinqhi_home)
        deleted = prune_quick_snapshots(keep=3, kinqhi_home=kinqhi_home)
        assert deleted == 7
        assert len(list_quick_snapshots(kinqhi_home=kinqhi_home)) == 3

    def test_snapshot_includes_pairing_directories(self, kinqhi_home):
        """Pairing JSONs live outside state.db — snapshot must capture them
        recursively (generic + per-platform) so approved-user lists survive
        disasters like #15733."""
        from kinqhi_cli.backup import create_quick_snapshot

        # Generic pairing store (new location)
        (kinqhi_home / "platforms" / "pairing").mkdir(parents=True)
        (kinqhi_home / "platforms" / "pairing" / "telegram-approved.json").write_text(
            '{"12345": {"user_name": "alice"}}'
        )
        (kinqhi_home / "platforms" / "pairing" / "discord-approved.json").write_text(
            '{"67890": {"user_name": "bob"}}'
        )
        # Legacy pairing store (old location)
        (kinqhi_home / "pairing").mkdir()
        (kinqhi_home / "pairing" / "matrix-approved.json").write_text(
            '{"@charlie:server": {"user_name": "charlie"}}'
        )
        # Feishu's separate JSON
        (kinqhi_home / "feishu_comment_pairing.json").write_text(
            '{"doc_abc": {"allow_from": ["user_xyz"]}}'
        )

        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        assert snap_id is not None

        snap_dir = kinqhi_home / "state-snapshots" / snap_id
        assert (snap_dir / "platforms" / "pairing" / "telegram-approved.json").exists()
        assert (snap_dir / "platforms" / "pairing" / "discord-approved.json").exists()
        assert (snap_dir / "pairing" / "matrix-approved.json").exists()
        assert (snap_dir / "feishu_comment_pairing.json").exists()

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        files = meta["files"]
        assert "platforms/pairing/telegram-approved.json" in files
        assert "platforms/pairing/discord-approved.json" in files
        assert "pairing/matrix-approved.json" in files
        assert "feishu_comment_pairing.json" in files

    def test_restore_recovers_pairing_data(self, kinqhi_home):
        """After restore, deleted pairing files reappear with original content."""
        from kinqhi_cli.backup import create_quick_snapshot, restore_quick_snapshot

        pairing_dir = kinqhi_home / "platforms" / "pairing"
        pairing_dir.mkdir(parents=True)
        approved = pairing_dir / "telegram-approved.json"
        approved.write_text('{"12345": {"user_name": "alice"}}')
        feishu = kinqhi_home / "feishu_comment_pairing.json"
        feishu.write_text('{"doc_abc": {"allow_from": ["user_xyz"]}}')

        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        assert snap_id is not None

        # Simulate the disaster — user loses both pairing files.
        approved.unlink()
        feishu.unlink()
        assert not approved.exists()
        assert not feishu.exists()

        assert restore_quick_snapshot(snap_id, kinqhi_home=kinqhi_home) is True
        assert approved.exists()
        assert '"alice"' in approved.read_text()
        assert feishu.exists()
        assert '"user_xyz"' in feishu.read_text()

    def test_empty_pairing_dir_does_not_fail(self, kinqhi_home):
        """An empty pairing directory should be silently skipped."""
        from kinqhi_cli.backup import create_quick_snapshot

        (kinqhi_home / "platforms" / "pairing").mkdir(parents=True)
        # Directory exists but contains no files.
        snap_id = create_quick_snapshot(kinqhi_home=kinqhi_home)
        # Other state still present → snapshot succeeds.
        assert snap_id is not None

# ---------------------------------------------------------------------------
# Pre-update backup (kinqhi update safety net)
# ---------------------------------------------------------------------------

class TestPreUpdateBackup:
    """Tests for create_pre_update_backup — the auto-backup ``kinqhi update``
    runs before touching anything."""

    @pytest.fixture
    def kinqhi_home(self, tmp_path):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        return root

    def test_creates_backup_under_backups_dir(self, kinqhi_home):
        from kinqhi_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        assert out.exists()
        assert out.parent == kinqhi_home / "backups"
        assert out.name.startswith("pre-update-")
        assert out.suffix == ".zip"

    def test_backup_contents_match_full_backup(self, kinqhi_home):
        """Pre-update backup should include the same user data that
        ``hermes backup`` would, and should exclude the same directories."""
        from kinqhi_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        # User data present
        assert "config.yaml" in names
        assert ".env" in names
        assert "sessions/abc123.json" in names
        assert "skills/my-skill/SKILL.md" in names
        assert "profiles/coder/config.yaml" in names
        # kinqhi repo excluded
        assert not any(n.startswith("kinqhi/") for n in names)
        # __pycache__ excluded
        assert not any("__pycache__" in n for n in names)
        # pid files excluded
        assert "gateway.pid" not in names

    def test_does_not_recurse_into_prior_backups(self, kinqhi_home):
        """The ``backups/`` directory must be excluded so that each backup
        doesn't grow exponentially by including all prior backups."""
        from kinqhi_cli.backup import create_pre_update_backup
        # First backup
        out1 = create_pre_update_backup(kinqhi_home=kinqhi_home)
        assert out1 is not None
        # Second backup — must not include the first
        out2 = create_pre_update_backup(kinqhi_home=kinqhi_home)
        assert out2 is not None
        with zipfile.ZipFile(out2) as zf:
            names = zf.namelist()
        assert not any(n.startswith("backups/") for n in names), (
            f"Pre-update backup recursed into backups/ — leaked: "
            f"{[n for n in names if n.startswith('backups/')]}"
        )

    def test_rotation_keeps_only_n(self, kinqhi_home):
        """After more than ``keep`` backups are created, older ones are
        pruned automatically."""
        import time as _t
        from kinqhi_cli.backup import create_pre_update_backup

        created = []
        for _ in range(5):
            out = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=3)
            created.append(out)
            _t.sleep(1.05)  # ensure distinct seconds in timestamp

        remaining = sorted(
            p.name for p in (kinqhi_home / "backups").iterdir()
            if p.name.startswith("pre-update-")
        )
        assert len(remaining) == 3
        # Oldest two should have been pruned
        assert created[0].name not in remaining
        assert created[1].name not in remaining
        # Newest three should remain
        assert created[4].name in remaining

    def test_rotation_preserves_manual_files(self, kinqhi_home):
        """Hand-dropped zips in ``backups/`` must not be touched by
        rotation — it only prunes files matching ``pre-update-*.zip``."""
        import time as _t
        from kinqhi_cli.backup import create_pre_update_backup

        (kinqhi_home / "backups").mkdir(exist_ok=True)
        manual = kinqhi_home / "backups" / "my-manual.zip"
        manual.write_bytes(b"manual backup")

        for _ in range(5):
            create_pre_update_backup(kinqhi_home=kinqhi_home, keep=2)
            _t.sleep(1.05)

        assert manual.exists(), "Manual backup zip was incorrectly pruned"

    def test_returns_none_if_root_missing(self, tmp_path):
        from kinqhi_cli.backup import create_pre_update_backup
        assert create_pre_update_backup(kinqhi_home=tmp_path / "does-not-exist") is None

    def test_keep_zero_does_not_delete_freshly_created_backup(self, kinqhi_home):
        """Regression: ``backup_keep: 0`` previously triggered ``backups[0:]``
        in the pruner — wiping the just-created zip and leaving the user
        with no recovery point.  The floor (keep>=1) preserves the new file
        regardless of misconfiguration; users who don't want backups should
        set ``pre_update_backup: false`` instead.
        """
        from kinqhi_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=0)
        assert out is not None
        assert out.exists(), (
            "keep=0 silently deleted the freshly-created backup; floor "
            "should preserve the just-written file."
        )

    def test_keep_negative_does_not_delete_freshly_created_backup(self, kinqhi_home):
        """Mirror coverage: any value <1 should be floored, not literally
        applied as a slice index."""
        from kinqhi_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=-3)
        assert out is not None
        assert out.exists()

    def test_keep_zero_still_prunes_older_backups(self, kinqhi_home):
        """The floor preserves the new backup but should NOT regress the
        rotation behaviour for older zips: a third call with keep=0 must
        still remove pre-existing backups beyond the (floored) limit of 1.
        """
        import time as _t
        from kinqhi_cli.backup import create_pre_update_backup

        first = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=5)
        _t.sleep(1.05)
        second = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=5)
        _t.sleep(1.05)
        third = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=0)

        remaining = {
            p.name for p in (kinqhi_home / "backups").iterdir()
            if p.name.startswith("pre-update-")
        }
        assert third.name in remaining, "Floor must preserve the new backup"
        assert first.name not in remaining and second.name not in remaining, (
            f"keep=0 floor of 1 should still prune older backups; "
            f"remaining={remaining}"
        )

    def test_skips_symlinked_files(self, kinqhi_home, tmp_path):
        """Pre-update backups must not dereference symlinks outside KINQHI_HOME."""
        from kinqhi_cli.backup import create_pre_update_backup

        outside = tmp_path / "outside-secret.txt"
        outside.write_text("outside secret\n")
        _symlink_file_or_skip(kinqhi_home / "skills" / "outside-link.txt", outside)

        out = create_pre_update_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "skills/outside-link.txt" not in names
            assert all(zf.read(name) != b"outside secret\n" for name in names)


class TestRunPreUpdateBackup:
    """Tests for the ``_run_pre_update_backup`` wrapper in main.py —
    covers config gate, ``--no-backup`` flag, and user-facing output."""

    @pytest.fixture
    def kinqhi_home(self, tmp_path, monkeypatch):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        # Point KINQHI_HOME at the temp dir so config + backup paths resolve here
        monkeypatch.setenv("KINQHI_HOME", str(root))
        # Make Path.home() point at tmp_path for anything that uses it
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Bust caches for kinqhi_cli.config + kinqhi_constants so they pick up KINQHI_HOME
        for mod in list(__import__("sys").modules.keys()):
            if mod.startswith("kinqhi_cli.config") or mod == "kinqhi_constants":
                del __import__("sys").modules[mod]
        return root

    def test_backup_flag_creates_backup(self, kinqhi_home, capsys):
        """--backup forces the pre-update backup for one run even when config is off."""
        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=False, backup=True))
        out = capsys.readouterr().out
        assert "Creating pre-update backup" in out
        assert "Saved:" in out
        assert "Restore:" in out
        assert "hermes import" in out
        assert "Disable:" in out
        # Actual backup was created
        backups = list((kinqhi_home / "backups").glob("pre-update-*.zip"))
        assert len(backups) == 1

    def test_default_enabled_creates_backup(self, kinqhi_home, capsys):
        """With the new safe default (``pre_update_backup: true``), every
        ``kinqhi update`` creates a backup before any destructive step
        runs — the cost is a few minutes of zip time vs. the alternative
        of silent total data loss of ``~/.kinqhi/`` observed in #48200
        when an update step computes a wrong path and the user had no
        safety net.
        """
        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert "Creating pre-update backup" in out
        assert "Saved:" in out
        backups = list((kinqhi_home / "backups").glob("pre-update-*.zip"))
        assert len(backups) == 1

    def test_no_backup_flag_skips(self, kinqhi_home, capsys):
        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=True, backup=False))
        out = capsys.readouterr().out
        assert "skipped (--no-backup)" in out
        assert "Creating pre-update backup" not in out
        # No backup written
        assert not (kinqhi_home / "backups").exists() or not list(
            (kinqhi_home / "backups").glob("pre-update-*.zip")
        )

    def test_config_enabled_creates_backup(self, kinqhi_home, capsys):
        """Users who explicitly set updates.pre_update_backup: true still get
        a backup on every update — this is the opt-in legacy behavior."""
        import yaml
        (kinqhi_home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": 22,
            "updates": {"pre_update_backup": True},
        }))
        import sys as _sys
        for mod in list(_sys.modules.keys()):
            if mod.startswith("kinqhi_cli.config"):
                del _sys.modules[mod]

        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert "Creating pre-update backup" in out
        assert "Saved:" in out
        backups = list((kinqhi_home / "backups").glob("pre-update-*.zip"))
        assert len(backups) == 1

    def test_config_disabled_is_silent(self, kinqhi_home, capsys):
        """Explicit pre_update_backup: false behaves the same as the default —
        silent no-op, no message spam."""
        import yaml
        (kinqhi_home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": 22,
            "updates": {"pre_update_backup": False},
        }))
        # Ensure config module re-reads
        import sys as _sys
        for mod in list(_sys.modules.keys()):
            if mod.startswith("kinqhi_cli.config"):
                del _sys.modules[mod]

        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert out == ""
        assert not list((kinqhi_home / "backups").glob("pre-update-*.zip")) \
            if (kinqhi_home / "backups").exists() else True

    def test_cli_flag_overrides_enabled_config(self, kinqhi_home, capsys):
        """--no-backup wins even when config says pre_update_backup: true."""
        import yaml
        (kinqhi_home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": 22,
            "updates": {"pre_update_backup": True},
        }))
        import sys as _sys
        for mod in list(_sys.modules.keys()):
            if mod.startswith("kinqhi_cli.config"):
                del _sys.modules[mod]

        from kinqhi_cli.main import _run_pre_update_backup
        _run_pre_update_backup(Namespace(no_backup=True, backup=False))
        out = capsys.readouterr().out
        assert "skipped (--no-backup)" in out


# ---------------------------------------------------------------------------
# Pre-migration backup (kinqhi claw migrate safety net)
# ---------------------------------------------------------------------------

class TestPreMigrationBackup:
    """Tests for create_pre_migration_backup — the auto-backup
    ``kinqhi claw migrate`` runs before mutating ~/.kinqhi/."""

    @pytest.fixture
    def kinqhi_home(self, tmp_path):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        return root

    def test_creates_backup_under_backups_dir(self, kinqhi_home):
        from kinqhi_cli.backup import create_pre_migration_backup
        out = create_pre_migration_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        assert out.exists()
        # Shares the backups/ directory with pre-update backups so `hermes
        # import` and the update-backup listing both pick them up.
        assert out.parent == kinqhi_home / "backups"
        assert out.name.startswith("pre-migration-")
        assert out.suffix == ".zip"

    def test_backup_uses_shared_exclusion_rules(self, kinqhi_home):
        """Pre-migration backup reuses the same exclusion rules as
        ``hermes backup`` / ``create_pre_update_backup`` — no drift."""
        from kinqhi_cli.backup import create_pre_migration_backup
        out = create_pre_migration_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        # User data present
        assert "config.yaml" in names
        assert ".env" in names
        assert "skills/my-skill/SKILL.md" in names
        # Same exclusions as the shared helper
        assert not any(n.startswith("kinqhi/") for n in names)
        assert not any("__pycache__" in n for n in names)
        assert "gateway.pid" not in names

    def test_restorable_with_hermes_import(self, kinqhi_home, tmp_path):
        """The zip produced by pre-migration backup must be a valid Kinqhi
        backup — `hermes import` should accept it."""
        from kinqhi_cli.backup import create_pre_migration_backup, _validate_backup_zip
        out = create_pre_migration_backup(kinqhi_home=kinqhi_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            valid, _reason = _validate_backup_zip(zf)
        assert valid, "pre-migration zip failed _validate_backup_zip"

    def test_does_not_recurse_into_prior_backups(self, kinqhi_home):
        from kinqhi_cli.backup import create_pre_migration_backup
        out1 = create_pre_migration_backup(kinqhi_home=kinqhi_home)
        assert out1 is not None
        out2 = create_pre_migration_backup(kinqhi_home=kinqhi_home)
        assert out2 is not None
        with zipfile.ZipFile(out2) as zf:
            names = zf.namelist()
        assert not any(n.startswith("backups/") for n in names)

    def test_rotation_keeps_only_n(self, kinqhi_home):
        import time as _t
        from kinqhi_cli.backup import create_pre_migration_backup

        created = []
        for _ in range(7):
            out = create_pre_migration_backup(kinqhi_home=kinqhi_home, keep=3)
            if out is not None:
                created.append(out)
            _t.sleep(1.05)  # timestamp resolution

        remaining = sorted((kinqhi_home / "backups").glob("pre-migration-*.zip"))
        assert len(remaining) <= 3, f"expected <=3 backups retained, got {len(remaining)}"

    def test_missing_kinqhi_home_returns_none(self, tmp_path):
        """Fresh install with no ~/.kinqhi yet — nothing to back up."""
        from kinqhi_cli.backup import create_pre_migration_backup
        missing = tmp_path / "does-not-exist"
        out = create_pre_migration_backup(kinqhi_home=missing)
        assert out is None

    def test_does_not_touch_pre_update_backups(self, kinqhi_home):
        """Pre-migration rotation must only prune pre-migration-*.zip files,
        leaving pre-update-*.zip backups untouched."""
        from kinqhi_cli.backup import create_pre_update_backup, create_pre_migration_backup
        update_backup = create_pre_update_backup(kinqhi_home=kinqhi_home, keep=5)
        assert update_backup is not None and update_backup.exists()
        # Spin up a lot of migration backups with keep=1
        import time as _t
        for _ in range(3):
            out = create_pre_migration_backup(kinqhi_home=kinqhi_home, keep=1)
            assert out is not None
            _t.sleep(1.05)
        # Update backup must still be there
        assert update_backup.exists(), "pre-migration rotation wrongly pruned the pre-update backup"


# ---------------------------------------------------------------------------
# Cron jobs auto-restore after silent migration loss (issue #34600)
# ---------------------------------------------------------------------------

class TestRestoreCronJobsIfEmptied:
    """`kinqhi update` config migration can leave cron/jobs.json valid-but-empty,
    silently dropping every scheduled job. `restore_cron_jobs_if_emptied` is the
    post-migration safety net that restores from the pre-update snapshot."""

    @staticmethod
    def _seed_jobs(path: Path, jobs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"jobs": jobs}))

    def _make_snapshot(self, kinqhi_home: Path, label="pre-update"):
        from kinqhi_cli.backup import create_quick_snapshot
        return create_quick_snapshot(label=label, kinqhi_home=kinqhi_home, keep=5)

    def test_restores_when_emptied_after_migration(self, tmp_path):
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        # Pre-update: 3 real jobs.
        self._seed_jobs(jobs_path, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        snap_id = self._make_snapshot(kinqhi_home)
        assert snap_id

        # Migration silently empties the file (valid JSON, zero jobs).
        jobs_path.write_text(json.dumps({"jobs": []}))

        result = restore_cron_jobs_if_emptied(snap_id, kinqhi_home=kinqhi_home)
        assert result is not None
        assert result["restored"] is True
        assert result["job_count"] == 3
        assert result["snapshot_id"] == snap_id

        # The live file now has the jobs back.
        restored = json.loads(jobs_path.read_text())
        assert len(restored["jobs"]) == 3

    def test_noop_when_live_file_still_has_jobs(self, tmp_path):
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [{"id": "a"}, {"id": "b"}])
        snap_id = self._make_snapshot(kinqhi_home)

        # Healthy path: file unchanged after update.
        result = restore_cron_jobs_if_emptied(snap_id, kinqhi_home=kinqhi_home)
        assert result is None

    def test_noop_when_snapshot_had_no_jobs(self, tmp_path):
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        # Pre-update genuinely had zero jobs; current is also empty.
        self._seed_jobs(jobs_path, [])
        snap_id = self._make_snapshot(kinqhi_home)
        jobs_path.write_text(json.dumps({"jobs": []}))

        result = restore_cron_jobs_if_emptied(snap_id, kinqhi_home=kinqhi_home)
        assert result is None

    def test_noop_when_live_file_unreadable(self, tmp_path):
        """An unparseable live file is left alone — that's a different failure
        mode the user should see, not silently overwrite."""
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [{"id": "a"}])
        snap_id = self._make_snapshot(kinqhi_home)
        jobs_path.write_text("{ this is not valid json")

        result = restore_cron_jobs_if_emptied(snap_id, kinqhi_home=kinqhi_home)
        assert result is None
        # File left untouched.
        assert jobs_path.read_text() == "{ this is not valid json"

    def test_noop_when_snapshot_id_missing(self, tmp_path):
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [])
        assert restore_cron_jobs_if_emptied(None, kinqhi_home=kinqhi_home) is None
        assert restore_cron_jobs_if_emptied("", kinqhi_home=kinqhi_home) is None

    def test_restores_legacy_bare_list_snapshot_shape(self, tmp_path):
        """A legacy snapshot storing a bare JSON list (not {"jobs": [...]}) is
        still counted and restored."""
        from kinqhi_cli.backup import restore_cron_jobs_if_emptied
        kinqhi_home = tmp_path / ".hermes"
        jobs_path = kinqhi_home / "cron" / "jobs.json"
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
        snap_id = self._make_snapshot(kinqhi_home)

        jobs_path.write_text(json.dumps({"jobs": []}))
        result = restore_cron_jobs_if_emptied(snap_id, kinqhi_home=kinqhi_home)
        assert result is not None
        assert result["job_count"] == 2
