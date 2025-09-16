# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import errno
import fcntl
import json
import os
import pathlib
import re
import signal
import sqlite3
from typing import Any

import cv2
import numpy as np
import torch


class FileLock:
    """
    A simple file lock implementation using fcntl.
    This is used to ensure that only one process can write to the cache at a time.

    Note: This will only work on Unix-like systems (Linux, macOS).
    It uses the `fcntl` module to lock a file for exclusive or shared access and uses a signal to handle timeouts.
    """

    def __init__(self, path: pathlib.Path, exclusive: bool = True, timeout_seconds=10):
        """
        Create a new file lock.

        Args:
            path (pathlib.Path): The path to the file to lock.
            exclusive (bool): If True, the lock will be exclusive (i.e., no other process can acquire the lock).
                If False, the lock will be shared (i.e., other processes can acquire the lock unless some other
                process has an exclusive lock on the file).
            timeout_seconds (int): The number of seconds to wait for the lock to be acquired before timing out.
                If the lock cannot be acquired within this time, a TimeoutError will be raised.
        """
        if os.name != "posix":
            raise NotImplementedError("File locking is only supported on Unix-like systems (Linux, macOS).")

        self._path = path
        self._lock_file_path = path
        if not self._lock_file_path.parent.exists():
            raise FileNotFoundError(f"Parent directory of lock file {self._lock_file_path.parent} does not exist. ")

        self._exclusive = exclusive
        self._timeout_seconds = timeout_seconds
        self._lock_fd = None

    @staticmethod
    def _timeout_signal_handler(signum, frame):
        """
        Signal handler callback for handling timeouts.

        Args:
            signum (int): The signal number (not used in this implementation).
            frame (frame): The current stack frame (not used in this implementation).
        """
        raise TimeoutError("File lock acquisition timed out.")

    def __enter__(self):
        """
        Acquire the file lock. This will block until the lock is acquired or a timeout occurs.

        Use this in a `with` statement to ensure the lock is released when done.

        _i.e._, use as follows:
        ```python
        with FileLock(path, exclusive=True):
            # Critical section of code that requires the lock
        ```

        Raises:
            TimeoutError: If the lock cannot be acquired within the specified timeout.
            NotImplementedError: If the filesystem does not support file locking.
        """
        os_open_flags = os.O_RDWR | os.O_TRUNC
        if not self._lock_file_path.exists():
            os_open_flags |= os.O_CREAT
        fd = os.open(self._lock_file_path, os_open_flags)

        signal.signal(signal.SIGALRM, self._timeout_signal_handler)
        signal.alarm(self._timeout_seconds)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH)
        except OSError as exception:
            os.close(fd)
            if exception.errno == errno.ENOSYS:
                raise NotImplementedError("FileSystem does not support flock") from exception
        else:
            self._lock_fd = fd
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def __exit__(self, exc_type, exc_value, traceback):
        fd = int(self._lock_fd) if self._lock_fd is not None else None
        if fd is None:
            return
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class SfmCache:
    """
    A simple SQLite-based dataset cache for storing and retrieving datasets.

    It allows for efficient storage and retrieval of datasets, including images,
    point clouds, and other data types.

    The cache is organized into folders which can contain files and subfolders.

    The cache is stored in a SQLite database, and the files are stored on disk in a directory structure.

    The cache supports the following data types:
    - jpg: JPEG images
    - png: PNG images
    - pt: PyTorch tensors
    - npy: NumPy arrays
    - json: JSON files
    - txt: Text files

    The cache is designed to be thread-safe and can be used by multiple processes concurrently.

    Internally, the cache uses a SQLite database to store metadata about the files and folders,
    and uses file locks to ensure that only one process can write to the cache at a time.

    The schema for the SQLite database is as follows:
    Tables:
    - metadata: Stores metadata about the cache, including the name, description, magic number, and version.
    - folders_{cache_id}: Stores information about folders in the cache, including their ID, parent ID, name, and description.
    - files_{cache_id}: Stores information about files in the cache, including their ID, name, data type, metadata, and folder ID.

    """

    magic_number = 0xAFAFAFAF  # Arbitrary magic number to identify the cache format
    version = "1.0.0"
    __SECRET__ = object()

    known_data_types = {
        "jpg",
        "png",
        "pt",
        "npy",
        "json",
        "txt",
    }

    def _validate_name(self, name: str, name_type: str) -> None:
        """
        Validate names for caches, files, folders, etc. in the cache.
        The name must be a nonempty string consisting only of alphanumeric characters and underscores.

        Args:
            name (str): The name to validate.
            name_type (str): The type of name being validated (e.g., "Cache", "File", "Directory").
                This is used for error messages to clarify what type of name is being validated.
        """
        if not isinstance(name, str):
            raise TypeError(f"{name_type} name must be a string, got {type(name)}")
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError(
                f"{name_type} name '{name}' contains invalid characters. "
                f"{name_type}s must only contain alphanumeric characters and underscores."
            )
        if len(name) == 0:
            raise ValueError(f"{name_type} name cannot be an empty string.")

    def __init__(
        self, db_path: pathlib.Path, cache_id: int, root_folder_id: int, current_folder_id: int, _private: Any = None
    ):
        """
        Create a new `Cache` instance with the specified database path, cache ID, root folder ID, and current folder ID.

        Note: You should not create a `Cache` instance directly. Instead, use the `Cache.get_cache()` method to create a cache
        or the `make_folder` method to create a folder within a cache.

        Args:
            db_path (pathlib.Path): The path to the SQLite database file for the cache.
            cache_id (int): The unique identifier for the cache in the database.
            root_folder_id (int): The unique ID of the root folder for this cache in the database.
            current_folder_id (int): The unique ID of the current folder for this cache in the database.
            _private (Any): A private parameter to prevent direct instantiation of the class.
                This should be set to `Cache.__SECRET__` when calling this constructor.
        """
        if _private != SfmCache.__SECRET__:
            raise RuntimeError(
                "Do not create a `Cache` instance directly. Instead use `Cache.get_cache()` to create a cache "
                "or make_folder to make a folder within a cache."
            )
        self._db_path = db_path
        self._cache_id = cache_id
        self._root_folder_id = root_folder_id
        self._current_folder_id = current_folder_id
        self._folders_table_name = f"folders_{self._cache_id}"
        self._files_table_name = f"files_{self._cache_id}"

        self.cache_root_path.mkdir(parents=True, exist_ok=True)
        self._file_lock_exclusive = FileLock(self.cache_root_path / "file_lock.lock", exclusive=True)
        self._file_lock_shared = FileLock(self.cache_root_path / "file_lock.lock", exclusive=False)

    @staticmethod
    def get_cache(cache_root: pathlib.Path, name: str, description: str) -> "SfmCache":
        """
        Get or create a new `Cache` with the given name and description.
        The name should be a nonempty string consisting only of alphanumeric characters and underscores.

        If a cache with the given name already exists, it will return the existing cache.

        Args:
            cache_root (pathlib.Path): The path to the root directory where the cache will be stored. This directory will be created if it does not exist.
            name (str): The name of the cache. Must be a nonempty string consisting only of alphanumeric characters and underscores.
            description (str): A description of the cache. This can be any string.

        Returns:
            Cache: An instance of `Cache` representing the cache.
        """
        db_path = cache_root / f"cache_{name}.db"
        if not cache_root.exists():
            cache_root.mkdir(parents=True, exist_ok=True)
        with FileLock(db_path.with_suffix(".lock")):
            cache_id, root_folder_id = SfmCache._initialize_database(db_path, name, description)

        return SfmCache(db_path, cache_id, root_folder_id, root_folder_id, _private=SfmCache.__SECRET__)

    @property
    def db_path(self) -> pathlib.Path:
        """
        Get the path to the SQLite database file.

        Returns:
            pathlib.Path: The path to the SQLite database file.
        """
        return self._db_path

    @property
    def cache_root_path(self) -> pathlib.Path:
        """
        Get the path to the directory where files in this cache are stored.

        Returns:
            pathlib.Path: The path to the root directory of the dataset cache.
        """
        return self._db_path.parent / f"cache_{self._cache_id}"

    @property
    def cache_id(self) -> int:
        """
        Get the unique ID of this cache in the database.

        Returns:
            int: The ID of the cache.
        """
        return self._cache_id

    @property
    def root_folder_id(self) -> int:
        """
        Get the unique ID of the root folder for this cache in the database.

        Returns:
            int: The unique ID of the root folder of this cache in the database.
        """
        return self._root_folder_id

    @property
    def current_folder_id(self) -> int:
        """
        Get the unique ID of the current folder for this cache in the database.

        Returns:
            int: The unique ID of the current folder of this cache in the database.
        """
        return self._current_folder_id

    @property
    def cache_name(self) -> str:
        """
        Get the name of the cache.

        Note: If this database manages multiple caches, the name is unique across those caches.

        Returns:
            str: The name of the cache.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM metadata WHERE id = ?", (self._cache_id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Cache with ID {self._cache_id} not found in the database.")
        finally:
            cursor.close()
            conn.close()

        return row[0]

    @property
    def cache_description(self) -> str:
        """
        Get the description of the cache.

        Returns:
            str: The description of the cache.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT description FROM metadata WHERE id = ?", (self._cache_id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Cache with ID {self._cache_id} not found in the database.")
        finally:
            cursor.close()
            conn.close()

        return row[0]

    @property
    def current_folder_name(self) -> str:
        """
        Get the name of the current folder which this cache reads/writes from/to.

        Returns:
            str: The name of the current folder.
        """
        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT name FROM {self._folders_table_name} WHERE id = ?", (self._current_folder_id,))
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"Directory with ID {self._current_folder_id} not found in the database.")
            finally:
                cursor.close()
                conn.close()

            return row[0]

    @property
    def current_folder_description(self) -> str:
        """
        Get the description of the current folder.

        Returns:
            str: The description of the current folder.
        """
        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SELECT description FROM {self._folders_table_name} WHERE id = ?", (self._current_folder_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"Directory with ID {self._current_folder_id} not found in the database.")
            finally:
                cursor.close()
                conn.close()

            return row[0]

    @property
    def current_folder_path(self) -> pathlib.Path:
        """
        Get the absolute path to the directory for the current folder of this cache.

        This is the directory where files for the current folder are stored.

        Returns:
            pathlib.Path: The absolute path to the directory for the current folder of this cache.
        """
        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                ret = self._get_path_for_folder(cursor, self._current_folder_id)
            finally:
                cursor.close()
                conn.close()

            return ret

    @property
    def num_folders(self) -> int:
        """
        Get the number of sub-folders in the current folder.

        Returns:
            int: The number of sub-folders in the current folder.
        """
        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:

                row = cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM {self._folders_table_name} WHERE parent_id = ?
                    """,
                    (self._current_folder_id,),
                )
                if row is None:
                    raise ValueError(f"Directory with ID {self._current_folder_id} not found in the database.")

                num_folders = row.fetchone()[0]
                if num_folders is None:
                    num_folders = 0
            finally:
                cursor.close()
                conn.close()

        return num_folders if row else 0

    @property
    def num_files(self) -> int:
        """
        Get the number of files in the current folder.

        Returns:
            int: The number of files in the current folder.
        """
        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM {self._files_table_name} WHERE folder_id = ?
                    """,
                    (self._current_folder_id,),
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
                conn.close()

            return row[0] if row else 0

    def has_folder(self, name: str) -> bool:
        """
        Return whether a folder with the given name exists in the current folder in this cache.

        Args:
            name (str): The name of the folder to check for existence.
                Must be a nonempty string consisting only of alphanumeric characters and underscores.
        Returns:
            bool: True if the folder exists, False otherwise.
        """
        self._validate_name(name, "Folder")

        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                row = cursor.execute(
                    f"""
                    SELECT id FROM {self._folders_table_name} WHERE name = ? AND parent_id = ?
                    """,
                    (name, self._current_folder_id),
                ).fetchone()
            finally:
                cursor.close()
                conn.close()

            return row is not None

    def has_file(self, name: str) -> bool:
        """
        Return whether a file with the given name exists in the current folder in this cache.

        Args:
            name (str): The name of the file to check for existence.
                Must be a nonempty string consisting only of alphanumeric characters and underscores.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        self._validate_name(name, "File")

        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                row = cursor.execute(
                    f"""
                    SELECT id FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (name, self._current_folder_id),
                ).fetchone()
            finally:
                cursor.close()
                conn.close()

            return row is not None

    def write_file(
        self,
        name: str,
        data: Any,
        data_type: str,
        metadata: dict = {},
        quality: int = 100,
    ):
        if data_type not in self.known_data_types:
            raise ValueError(
                f"Unknown data type {data_type} for property {name}. Must be one of {self.known_data_types}"
            )
        self._validate_name(name, "File")

        with self._file_lock_exclusive:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            try:
                metadata_json = json.dumps(metadata)

                # Insert or update the file entry in the database
                row = cursor.execute(
                    f"""
                    SELECT id FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (name, self._current_folder_id),
                ).fetchone()
                if row is None:
                    cursor.execute(
                        f"""
                        INSERT INTO {self._files_table_name} (name, data_type, metadata, folder_id, cache_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (name, data_type, metadata_json, self._current_folder_id, self._cache_id),
                    )
                else:
                    file_id = row[0]
                    cursor.execute(
                        f"""
                        UPDATE {self._files_table_name}
                        SET data_type = ?, metadata = ?
                        WHERE id = ? AND folder_id = ?
                        """,
                        (data_type, metadata_json, file_id, self._current_folder_id),
                    )

                file_path = self._get_path_for_folder(cursor, self._current_folder_id) / f"{name}.{data_type}"
                if data_type == "jpg":
                    cv2.imwrite(str(file_path), data, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif data_type == "png":
                    cv2.imwrite(str(file_path), data)
                elif data_type == "pt":
                    torch.save(data, file_path)
                elif data_type == "npy":
                    np.save(file_path, data)
                elif data_type == "json":
                    with open(file_path, "w") as f:
                        json.dump(data, f)
                elif data_type == "txt":
                    with open(file_path, "w") as f:
                        f.write(data)
                else:
                    raise ValueError(f"Unknown data type {data_type} for property {name}")

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Failed to write file '{name}'") from e
            finally:
                cursor.close()
                conn.close()

            return {
                "data_type": data_type,
                "metadata": metadata,
                "path": file_path,
            }

    def read_file(self, name: str) -> tuple[dict[str, Any], Any]:
        self._validate_name(name, "File")

        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"""
                    SELECT data_type, metadata FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (name, self._current_folder_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"File '{name}' not found in the current folder.")

                data_type, metadata_json = row
                metadata = json.loads(metadata_json)

                file_path = self._get_path_for_folder(cursor, self._current_folder_id) / f"{name}.{data_type}"
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"File '{name}' not found at path {file_path}. The cache is likely corrupted."
                    )

                if data_type == "jpg" or data_type == "png":
                    data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                elif data_type == "pt":
                    data = torch.load(file_path, weights_only=False)
                elif data_type == "npy":
                    data = np.load(file_path)
                elif data_type == "json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                elif data_type == "txt":
                    with open(file_path, "r") as f:
                        data = f.read()
                else:
                    raise ValueError(
                        f"Unknown data type {data_type} for image property. Must be one of {SfmCache.known_data_types}"
                    )
            finally:
                cursor.close()
                conn.close()

        return metadata, data

    def delete_file(self, key: str) -> None:
        self._validate_name(key, "File")

        with self._file_lock_exclusive:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(
                    f"""
                    SELECT data_type FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (key, self._current_folder_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"File '{key}' not found in the current folder.")

                data_type = row[0]

                cursor.execute(
                    f"""
                    DELETE FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (key, self._current_folder_id),
                )

                file_path = self._get_path_for_folder(cursor, self._current_folder_id) / f"{key}.{data_type}"
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
                else:
                    raise FileNotFoundError(
                        f"File '{key}' not found at path {file_path}. The cache is likely corrupted."
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Failed to delete file '{key}': {e}")
            finally:
                cursor.close()
                conn.close()

    def get_file_metadata(self, key: str) -> dict[str, Any]:
        self._validate_name(key, "File")

        with self._file_lock_shared:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"""
                    SELECT data_type, metadata FROM {self._files_table_name} WHERE name = ? AND folder_id = ?
                    """,
                    (key, self._current_folder_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"File '{key}' not found in the current folder.")

                data_type, metadata_json = row
                metadata = json.loads(metadata_json)
                file_path = self._get_path_for_folder(cursor, self._current_folder_id) / f"{key}.{data_type}"
            finally:
                cursor.close()
                conn.close()

        return {
            "data_type": data_type,
            "metadata": metadata,
            "path": file_path,
        }

    def make_folder(self, name: str, description: str = "") -> "SfmCache":
        self._validate_name(name, "Folder")

        with self._file_lock_exclusive:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            try:
                row = cursor.execute(
                    f"""
                    SELECT id FROM {self._folders_table_name} WHERE name = ? AND parent_id = ?
                    """,
                    (name, self._current_folder_id),
                ).fetchone()

                if row is None:
                    cursor.execute(
                        f"""
                        INSERT INTO {self._folders_table_name} (cache_id, parent_id, name, description)
                        VALUES (?, ?, ?, ?)
                        """,
                        (self._cache_id, self._current_folder_id, name, description),
                    )

                row = cursor.execute(
                    f"""
                    SELECT id FROM {self._folders_table_name} WHERE name = ? AND parent_id = ?
                    """,
                    (name, self._current_folder_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to create new folder in the database.")
                new_folder_id = row[0]
                new_folder_path = self._get_path_for_folder(cursor, new_folder_id)
                new_folder_path.mkdir(parents=True, exist_ok=True)

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise ValueError(f"Failed to create '{name}' in current folder.") from e
            finally:
                cursor.close()
                conn.close()
            return SfmCache(
                db_path=self._db_path,
                cache_id=self._cache_id,
                root_folder_id=self._root_folder_id,
                current_folder_id=new_folder_id,
                _private=SfmCache.__SECRET__,
            )

    def clear_current_folder(self) -> None:
        """
        Clear all data in the current folder, including all subfolders and files.
        """
        with self._file_lock_exclusive:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    DELETE FROM {self._folders_table_name} WHERE id = ? AND cache_id = ?""",
                    (self._current_folder_id, self._cache_id),
                )
                cursor.execute(
                    f"""
                    DELETE FROM {self._files_table_name} WHERE folder_id = ? AND cache_id = ?""",
                    (self._current_folder_id, self._cache_id),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to clear cache folder {self.current_folder_name}") from e
            finally:
                cursor.close()
                conn.close()

    def _get_path_for_folder(self, cursor: sqlite3.Cursor, folder_id: int) -> pathlib.Path:
        """
        Get the absolute path corresponding to the given folder ID in the database.

        Args:
            cursor (sqlite3.Cursor): The SQLite cursor to use for database queries.
            folder_id (int): The ID of the folder for which to get the path.
        Returns:
            pathlib.Path: The absolute path to the current folder for this cache.
        """

        path_components = []
        current_folder_id = folder_id
        while current_folder_id is not None:
            row = cursor.execute(
                f"SELECT name, parent_id FROM {self._folders_table_name} WHERE id = ? AND cache_id = ?",
                (current_folder_id, self._cache_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Directory with ID {current_folder_id} not found in the database.")
            path_components.append(row[0])
            current_folder_id = row[1]

        if len(path_components) == 0:
            raise ValueError(f"No folders found in the folder for ID {folder_id}.")

        if path_components[-1] != "root":
            raise ValueError("Folder path does not end with 'root'. This is unexpected.")

        path_components.reverse()
        path_components = path_components[1:]  # Skip the root folder name
        return self.cache_root_path / pathlib.Path(*path_components)

    @staticmethod
    def _initialize_database(db_path: pathlib.Path, name: str, description: str) -> tuple[int, int]:
        """
        Initialize the SQLite database with the necessary tables.
        This should be called once when the cache is created.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    magic INT NOT NULL,
                    version TEXT NOT NULL
                )
                """
            )
            cache_id_or_none = cursor.execute(
                """
                SELECT id FROM metadata WHERE name = ?
                """,
                (name,),
            ).fetchone()
            if cache_id_or_none is None:
                cache_id = cursor.execute(
                    """
                    INSERT INTO metadata (name, description, magic, version)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        description,
                        SfmCache.magic_number,
                        SfmCache.version,
                    ),
                ).lastrowid
            else:
                cache_id = cache_id_or_none[0]

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS folders_{cache_id} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_id INTEGER NOT NULL,
                    parent_id INTEGER,
                    name TEXT NOT NULL,
                    description TEXT,
                    FOREIGN KEY (cache_id) REFERENCES metadata (id),
                    FOREIGN KEY (parent_id) REFERENCES folders_{cache_id} (id)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS files_{cache_id} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    cache_id INTEGER NOT NULL,
                    data_type TEXT NOT NULL,
                    metadata BLOB,
                    folder_id INTEGER,
                    FOREIGN KEY (folder_id) REFERENCES folders_{cache_id} (id)
                    FOREIGN KEY (cache_id) REFERENCES metadata (id)
                )
                """
            )

            row = cursor.execute(
                f"""
                SELECT id FROM folders_{cache_id} WHERE parent_id IS NULL AND name = ?
                """,
                ("root",),
            ).fetchone()
            if row is None:
                root_folder_id = cursor.execute(
                    f"""
                    INSERT INTO folders_{cache_id} (cache_id, parent_id, name, description)
                    VALUES (?, NULL, ?, ?)
                    """,
                    (cache_id, "root", "Root folder for the cache"),
                ).lastrowid
            else:
                root_folder_id = row[0]

            if not isinstance(root_folder_id, int):
                raise TypeError(f"Root folder ID must be an integer, got {type(root_folder_id)}")
            if not isinstance(cache_id, int):
                raise TypeError(f"Cache ID must be an integer, got {type(cache_id)}")
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to initialize database at {db_path}: {e}") from e
        finally:
            cursor.close()
            conn.close()

        return cache_id, root_folder_id
