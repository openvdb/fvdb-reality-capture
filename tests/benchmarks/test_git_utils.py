# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for git utilities in benchmark_utils._common.
"""

import pathlib
import subprocess
import tempfile
import unittest


class TestGetGitInfo(unittest.TestCase):
    """Tests for get_git_info function."""

    def setUp(self):
        """Set up a temporary git repository for testing."""
        import sys

        # Add the comparative directory to path for imports
        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import get_git_info

        self.get_git_info = get_git_info

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )

        # Create initial commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("test content")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_git_info_basic(self):
        """Test basic git info retrieval."""
        info = self.get_git_info(self.repo_path)

        self.assertIsNotNone(info["commit"])
        self.assertEqual(len(info["commit"]), 40)  # Full SHA is 40 chars
        self.assertIsNotNone(info["short_commit"])
        self.assertEqual(len(info["short_commit"]), 7)
        self.assertEqual(info["dirty"], False)
        self.assertEqual(str(info["path"]), str(self.repo_path))

    def test_get_git_info_dirty(self):
        """Test detection of dirty working directory."""
        # Modify file without committing
        test_file = self.repo_path / "test.txt"
        test_file.write_text("modified content")

        info = self.get_git_info(self.repo_path)

        self.assertEqual(info["dirty"], True)

    def test_get_git_info_branch(self):
        """Test branch detection."""
        # Should be on main/master branch after init
        info = self.get_git_info(self.repo_path)

        # Branch could be 'main' or 'master' depending on git config
        self.assertIn(info["branch"], ["main", "master"])

    def test_get_git_info_nonexistent_path(self):
        """Test handling of non-existent path."""
        info = self.get_git_info(pathlib.Path("/nonexistent/path"))

        self.assertIsNone(info["commit"])
        self.assertIn("error", info)

    def test_get_git_info_not_a_repo(self):
        """Test handling of path that is not a git repo."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            info = self.get_git_info(pathlib.Path(tmpdir))

            self.assertIsNone(info["commit"])
            self.assertIn("error", info)


class TestGetCurrentCommit(unittest.TestCase):
    """Tests for get_current_commit function."""

    def setUp(self):
        """Set up a temporary git repository for testing."""
        import sys

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import get_current_commit

        self.get_current_commit = get_current_commit

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo with a commit
        subprocess.run(["git", "init"], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )

        test_file = self.repo_path / "test.txt"
        test_file.write_text("test content")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_current_commit(self):
        """Test getting current commit."""
        commit = self.get_current_commit(self.repo_path)

        self.assertIsNotNone(commit)
        self.assertEqual(len(commit), 40)

    def test_get_current_commit_nonexistent(self):
        """Test getting commit from non-existent path."""
        commit = self.get_current_commit(pathlib.Path("/nonexistent/path"))

        self.assertIsNone(commit)


class TestCheckoutCommit(unittest.TestCase):
    """Tests for checkout_commit function."""

    def setUp(self):
        """Set up a temporary git repository with multiple commits."""
        import sys

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import checkout_commit, get_current_commit

        self.checkout_commit = checkout_commit
        self.get_current_commit = get_current_commit

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )

        # Create first commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("version 1")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "First commit"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        self.first_commit = self.get_current_commit(self.repo_path)

        # Create second commit
        test_file.write_text("version 2")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Second commit"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        )
        self.second_commit = self.get_current_commit(self.repo_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkout_commit(self):
        """Test checking out a specific commit."""
        # We should be at second commit
        self.assertEqual(self.get_current_commit(self.repo_path), self.second_commit)

        # Checkout first commit
        result = self.checkout_commit(self.repo_path, self.first_commit)

        self.assertTrue(result)
        self.assertEqual(self.get_current_commit(self.repo_path), self.first_commit)

        # Verify file content changed
        test_file = self.repo_path / "test.txt"
        self.assertEqual(test_file.read_text(), "version 1")

    def test_checkout_invalid_commit(self):
        """Test checking out an invalid commit."""
        result = self.checkout_commit(self.repo_path, "invalid_commit_sha")

        self.assertFalse(result)


class TestGetCommitsFromOptConfig(unittest.TestCase):
    """Tests for get_commits_from_opt_config function."""

    def setUp(self):
        """Set up imports."""
        import sys

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from comparison_benchmark import get_commits_from_opt_config

        self.get_commits_from_opt_config = get_commits_from_opt_config

    def test_no_commits_section(self):
        """Test opt_config without commits section."""
        opt_config = {
            "framework": "fvdb",
            "name": "test",
        }

        commits = self.get_commits_from_opt_config(opt_config)

        self.assertIsNone(commits["fvdb_core"])
        self.assertIsNone(commits["fvdb_reality_capture"])
        self.assertIsNone(commits["gsplat"])

    def test_with_commits_section(self):
        """Test opt_config with commits section."""
        opt_config = {
            "framework": "fvdb",
            "commits": {
                "fvdb_core": "abc123",
                "fvdb_reality_capture": "def456",
            },
            "name": "test",
        }

        commits = self.get_commits_from_opt_config(opt_config)

        self.assertEqual(commits["fvdb_core"], "abc123")
        self.assertEqual(commits["fvdb_reality_capture"], "def456")
        self.assertIsNone(commits["gsplat"])

    def test_partial_commits(self):
        """Test opt_config with only some commits specified."""
        opt_config = {
            "framework": "fvdb",
            "commits": {
                "fvdb_core": "abc123",
            },
            "name": "test",
        }

        commits = self.get_commits_from_opt_config(opt_config)

        self.assertEqual(commits["fvdb_core"], "abc123")
        self.assertIsNone(commits["fvdb_reality_capture"])


class TestGetCommitKey(unittest.TestCase):
    """Tests for get_commit_key function."""

    def setUp(self):
        """Set up imports."""
        import sys

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from comparison_benchmark import get_commit_key

        self.get_commit_key = get_commit_key

    def test_no_commits(self):
        """Test commit key with no commits."""
        opt_config = {"framework": "fvdb"}

        key = self.get_commit_key(opt_config)

        self.assertEqual(key, (None, None, None))

    def test_with_commits(self):
        """Test commit key with commits."""
        opt_config = {
            "framework": "fvdb",
            "commits": {
                "fvdb_core": "abc123",
                "fvdb_reality_capture": "def456",
            },
        }

        key = self.get_commit_key(opt_config)

        self.assertEqual(key, ("abc123", "def456", None))

    def test_commit_key_hashable(self):
        """Test that commit key is hashable and can be used as dict key."""
        opt_config = {
            "framework": "fvdb",
            "commits": {
                "fvdb_core": "abc123",
            },
        }

        key = self.get_commit_key(opt_config)

        # Should be usable as dict key
        d = {key: "test_value"}
        self.assertEqual(d[key], "test_value")


if __name__ == "__main__":
    unittest.main()
