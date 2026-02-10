# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for git utilities in benchmark_utils._common.
"""

import pathlib
import tempfile
import unittest

from git import Repo


class TestGetGitInfo(unittest.TestCase):
    """Tests for get_git_info function."""

    def setUp(self):
        """Set up a temporary git repository for testing."""
        import sys

        self._original_sys_path = sys.path[:]

        # Add the comparative directory to path for imports
        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import get_git_info

        self.get_git_info = get_git_info

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo using GitPython
        repo = Repo.init(self.repo_path)
        with repo.config_writer() as config:
            config.set_value("user", "email", "test@test.com")
            config.set_value("user", "name", "Test User")

        # Create initial commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("test content")
        repo.index.add(["test.txt"])
        repo.index.commit("Initial commit")

    def tearDown(self):
        """Clean up temporary directory and restore sys.path."""
        import shutil
        import sys

        sys.path[:] = self._original_sys_path

        # Safety check: ensure temp_dir is within system temp directory
        temp_dir_resolved = pathlib.Path(self.temp_dir).resolve()
        system_temp = pathlib.Path(tempfile.gettempdir()).resolve()

        if not str(temp_dir_resolved).startswith(str(system_temp)):
            raise RuntimeError(
                f"Refusing to delete {temp_dir_resolved}: not within system temp directory {system_temp}"
            )

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
        info = self.get_git_info(self.repo_path)

        # Compare against the actual branch name from the repo rather than
        # hardcoding, since the default initial branch is configurable.
        repo = Repo(self.repo_path)
        expected_branch = repo.active_branch.name
        self.assertEqual(info["branch"], expected_branch)

    def test_get_git_info_nonexistent_path(self):
        """Test handling of non-existent path."""
        info = self.get_git_info(pathlib.Path("/nonexistent/path"))

        self.assertIsNone(info["commit"])
        self.assertIn("error", info)

    def test_get_git_info_not_a_repo(self):
        """Test handling of path that is not a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = self.get_git_info(pathlib.Path(tmpdir))

            self.assertIsNone(info["commit"])
            self.assertIn("error", info)


class TestGetCurrentCommit(unittest.TestCase):
    """Tests for get_current_commit function."""

    def setUp(self):
        """Set up a temporary git repository for testing."""
        import sys

        self._original_sys_path = sys.path[:]

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import get_current_commit

        self.get_current_commit = get_current_commit

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo using GitPython
        repo = Repo.init(self.repo_path)
        with repo.config_writer() as config:
            config.set_value("user", "email", "test@test.com")
            config.set_value("user", "name", "Test User")

        # Create initial commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("test content")
        repo.index.add(["test.txt"])
        repo.index.commit("Initial commit")

    def tearDown(self):
        """Clean up temporary directory and restore sys.path."""
        import shutil
        import sys

        sys.path[:] = self._original_sys_path

        # Safety check: ensure temp_dir is within system temp directory
        temp_dir_resolved = pathlib.Path(self.temp_dir).resolve()
        system_temp = pathlib.Path(tempfile.gettempdir()).resolve()

        if not str(temp_dir_resolved).startswith(str(system_temp)):
            raise RuntimeError(
                f"Refusing to delete {temp_dir_resolved}: not within system temp directory {system_temp}"
            )

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

        self._original_sys_path = sys.path[:]

        comparative_dir = pathlib.Path(__file__).parent / "comparative"
        sys.path.insert(0, str(comparative_dir))

        from benchmark_utils._common import checkout_commit, get_current_commit

        self.checkout_commit = checkout_commit
        self.get_current_commit = get_current_commit

        # Create a temporary directory with a git repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = pathlib.Path(self.temp_dir)

        # Initialize git repo using GitPython
        repo = Repo.init(self.repo_path)
        with repo.config_writer() as config:
            config.set_value("user", "email", "test@test.com")
            config.set_value("user", "name", "Test User")

        # Create first commit
        test_file = self.repo_path / "test.txt"
        test_file.write_text("version 1")
        repo.index.add(["test.txt"])
        first_commit_obj = repo.index.commit("First commit")
        self.first_commit = first_commit_obj.hexsha

        # Create second commit
        test_file.write_text("version 2")
        repo.index.add(["test.txt"])
        second_commit_obj = repo.index.commit("Second commit")
        self.second_commit = second_commit_obj.hexsha

    def tearDown(self):
        """Clean up temporary directory and restore sys.path."""
        import shutil
        import sys

        sys.path[:] = self._original_sys_path

        # Safety check: ensure temp_dir is within system temp directory
        temp_dir_resolved = pathlib.Path(self.temp_dir).resolve()
        system_temp = pathlib.Path(tempfile.gettempdir()).resolve()

        if not str(temp_dir_resolved).startswith(str(system_temp)):
            raise RuntimeError(
                f"Refusing to delete {temp_dir_resolved}: not within system temp directory {system_temp}"
            )

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

    def test_checkout_dirty_repo_raises_error(self):
        """Test that checkout fails with RuntimeError if repo has uncommitted changes."""
        # Make the repo dirty by modifying a file without committing
        test_file = self.repo_path / "test.txt"
        test_file.write_text("uncommitted changes")

        # Attempting to checkout a different commit should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.checkout_commit(self.repo_path, self.first_commit)

        self.assertIn("uncommitted changes", str(context.exception))
        self.assertIn("commit or stash", str(context.exception))


if __name__ == "__main__":
    unittest.main()
