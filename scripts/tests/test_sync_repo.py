from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "sync_repo.py"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL, text=True).strip()


class GitSynchronization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin.git"
        self.pc, self.server = self.root / "pc", self.root / "server"
        subprocess.run(["git", "init", "--bare", str(self.origin)], check=True, capture_output=True)
        subprocess.run(["git", "clone", str(self.origin), str(self.pc)], check=True, capture_output=True)
        self.identity(self.pc)
        git(self.pc, "checkout", "-b", "main")
        (self.pc / "code.py").write_text("value = 1\n")
        git(self.pc, "add", "code.py")
        git(self.pc, "commit", "-m", "Initial source")
        git(self.pc, "push", "-u", "origin", "main")
        subprocess.run(["git", "clone", "--branch", "main", str(self.origin), str(self.server)], check=True, capture_output=True)
        self.identity(self.server)

    @staticmethod
    def identity(path):
        git(path, "config", "user.name", "Sync test")
        git(path, "config", "user.email", "sync-test@example.invalid")

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, checkout, action):
        return subprocess.run([sys.executable, str(SCRIPT), action, "--repo-root", str(checkout)], capture_output=True, text=True)

    def new_commit(self, checkout, filename="code.py", content="value = 2\n"):
        (checkout / filename).write_text(content)
        git(checkout, "add", filename)
        git(checkout, "commit", "-m", "Update source")

    def test_pc_origin_server_fast_forward_preserves_ignored_data(self):
        (self.server / ".git/info/exclude").write_text("local_artifacts/\n")
        artifact = self.server / "local_artifacts/weight.pt"
        artifact.parent.mkdir()
        artifact.write_bytes(b"local-only artifact")
        self.new_commit(self.pc)
        self.assertEqual(self.invoke(self.pc, "push").returncode, 0)
        self.assertEqual(self.invoke(self.server, "pull").returncode, 0)
        self.assertEqual(git(self.pc, "rev-parse", "HEAD"), git(self.server, "rev-parse", "HEAD"))
        self.assertEqual(artifact.read_bytes(), b"local-only artifact")

    def test_dirty_server_is_not_overwritten(self):
        self.new_commit(self.pc)
        git(self.pc, "push", "origin", "main")
        previous = git(self.server, "rev-parse", "HEAD")
        (self.server / "code.py").write_text("valuable local edit\n")
        result = self.invoke(self.server, "pull")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Uncommitted", result.stderr)
        self.assertEqual(git(self.server, "rev-parse", "HEAD"), previous)
        self.assertEqual((self.server / "code.py").read_text(), "valuable local edit\n")

    def test_divergent_commits_are_not_force_pushed(self):
        self.new_commit(self.pc)
        git(self.pc, "push", "origin", "main")
        self.new_commit(self.server, "server.txt", "server-only commit\n")
        local_before = git(self.server, "rev-parse", "HEAD")
        remote_before = git(self.origin, "rev-parse", "refs/heads/main")
        result = self.invoke(self.server, "push")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(git(self.server, "rev-parse", "HEAD"), local_before)
        self.assertEqual(git(self.origin, "rev-parse", "refs/heads/main"), remote_before)


if __name__ == "__main__":
    unittest.main()
