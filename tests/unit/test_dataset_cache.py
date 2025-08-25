# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import functools
import multiprocessing
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

import numpy as np
import torch
from datasets.dataset_cache import DatasetCache


class BasicDatasetCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache_name = "test_cache"
        self.cache_description = "A test cache for unit tests"
        self.cache_root = pathlib.Path(tempfile.mkdtemp(prefix="dataset_cache_test_"))

        self.db_path = self.cache_root / f"cache_{self.cache_name}.db"
        if self.db_path.exists():
            self.db_path.unlink()

        self.cache = DatasetCache.get_cache(
            name=self.cache_name,
            description=self.cache_description,
            cache_root=self.cache_root,
        )

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)

    def test_cache_initialization(self):
        self.assertEqual(self.cache.cache_name, self.cache_name)
        self.assertEqual(self.cache.cache_description, self.cache_description)
        self.assertEqual(self.cache.db_path, self.db_path)
        self.assertEqual(self.cache.current_folder_name, "root")

    def test_file_io(self):
        num_files_added = 0
        for data_type in ["png", "jpg", "npy", "pt", "txt", "json"]:
            file_name = f"test_file_{data_type}"

            if data_type in ("png", "jpg", "npy"):
                img = np.random.rand(100, 100, 3) * 255
                data = img.astype(np.uint8)
            elif data_type == "pt":
                img = torch.rand(100, 100, 3)
                data = {"tensor": img, "metadata": {"width": 100, "height": 100, "channels": 3}}
            elif data_type == "txt":
                data = "This is a test text file."
            elif data_type == "json":
                data = {"key": "value", "width": 100, "height": 100, "channels": 3}

            self.cache.write_file(
                name=file_name, data=data, data_type=data_type, metadata={"width": 100, "height": 100, "channels": 3}
            )
            num_files_added += 1

            self.assertTrue(self.cache.has_file(file_name))
            self.assertEqual(self.cache.num_files, num_files_added)
            self.assertEqual(self.cache.num_folders, 0)

            read_meta, read_data = self.cache.read_file(file_name)
            self.assertIsNotNone(read_data)
            self.assertIsNotNone(read_meta)

            if data_type in ("png", "jpg", "npy"):
                self.assertEqual(read_data.shape, (100, 100, 3))
                self.assertEqual(read_meta["width"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)
                assert isinstance(data, np.ndarray)
                assert isinstance(read_data, np.ndarray)
                if data_type != "jpg":
                    self.assertTrue(np.allclose(data, read_data))
            elif data_type == "pt":
                self.assertEqual(read_data["tensor"].shape, (100, 100, 3))
                self.assertEqual(read_meta["width"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)
                assert isinstance(data, dict)
                assert isinstance(read_data, dict)
                self.assertIn("tensor", read_data)
                self.assertIn("metadata", read_data)
                assert isinstance(data["tensor"], torch.Tensor)
                self.assertEqual(data["metadata"], read_data["metadata"])
                self.assertTrue(torch.allclose(data["tensor"], read_data["tensor"]))
            elif data_type == "txt":
                self.assertEqual(read_data, data)
                self.assertEqual(read_meta["width"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)
            elif data_type == "json":
                self.assertEqual(read_data, data)
                self.assertEqual(read_meta["width"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)

    def test_writing_updates(self):
        file_name = "test_file_update"
        initial_data = np.random.rand(100, 100, 3) * 255
        initial_data = initial_data.astype(np.uint8)
        self.cache.write_file(
            name=file_name,
            data=initial_data.astype(np.uint8),
            data_type="png",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read_meta, read_data = self.cache.read_file(file_name)
        self.assertTrue(np.allclose(initial_data, read_data))
        self.assertEqual(read_meta["width"], 100)
        self.assertEqual(read_meta["height"], 100)
        self.assertEqual(read_meta["channels"], 3)

        updated_data = np.random.rand(10, 10, 3) * 255
        updated_data = updated_data.astype(np.uint8)
        self.cache.write_file(
            name=file_name,
            data=updated_data,
            data_type="npy",
            metadata={"width": 10, "height": 10, "channels": 3},
        )

        read_meta, read_data = self.cache.read_file(file_name)
        self.assertTrue(np.allclose(updated_data, read_data))
        self.assertEqual(read_meta["width"], 10)
        self.assertEqual(read_meta["height"], 10)
        self.assertEqual(read_meta["channels"], 3)

    def test_subfolders(self):
        subfolder_name = "subfolder"
        subfolder = self.cache.make_folder(subfolder_name, description="A test subfolder")
        self.assertEqual(self.cache.current_folder_name, "root")
        self.assertEqual(subfolder.current_folder_name, subfolder_name)
        self.assertEqual(subfolder.current_folder_description, "A test subfolder")

        file_name = "test_file_in_folder"
        data = np.random.rand(100, 100, 3) * 255
        data = data.astype(np.uint8)
        self.cache.write_file(
            name=file_name,
            data=data,
            data_type="png",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read_meta, read_data = self.cache.read_file(file_name)
        self.assertTrue(np.allclose(data, read_data))
        self.assertEqual(read_meta["width"], 100)
        self.assertEqual(read_meta["height"], 100)
        self.assertEqual(read_meta["channels"], 3)

        self.assertTrue(self.cache.has_file(file_name))
        self.assertTrue(self.cache.has_folder(subfolder_name))
        self.assertEqual(self.cache.num_files, 1)
        self.assertEqual(self.cache.num_folders, 1)

        self.assertEqual(subfolder.num_files, 0)
        self.assertEqual(subfolder.num_folders, 0)

        # Different name same folder
        sub_file_name = "test_file_in_subfolder"
        sub_data1 = np.random.rand(100, 100, 3) * 255
        sub_data1 = data.astype(np.uint8)
        subfolder.write_file(
            name=sub_file_name,
            data=sub_data1,
            data_type="npy",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read_sub_meta1, read_sub_data1 = subfolder.read_file(sub_file_name)
        self.assertTrue(np.allclose(sub_data1, read_sub_data1))
        self.assertEqual(read_sub_meta1["width"], 100)
        self.assertEqual(read_sub_meta1["height"], 100)
        self.assertEqual(read_sub_meta1["channels"], 3)

        self.assertTrue(subfolder.has_file(sub_file_name))
        self.assertFalse(self.cache.has_file(sub_file_name))

        self.assertEqual(self.cache.num_files, 1)
        self.assertEqual(self.cache.num_folders, 1)

        self.assertEqual(subfolder.num_files, 1)
        self.assertEqual(subfolder.num_folders, 0)

        # Same name, different folder
        sub_data2 = np.random.rand(100, 100, 3) * 255
        sub_data2 = sub_data2.astype(np.uint8)
        subfolder.write_file(
            name=file_name,
            data=sub_data2,
            data_type="npy",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read_sub_meta2, read_sub_data2 = subfolder.read_file(file_name)
        self.assertTrue(np.allclose(sub_data2, read_sub_data2))
        self.assertEqual(read_sub_meta2["width"], 100)
        self.assertEqual(read_sub_meta2["height"], 100)
        self.assertEqual(read_sub_meta2["channels"], 3)

        self.assertTrue(subfolder.has_file(file_name))
        self.assertTrue(subfolder.has_file(sub_file_name))
        self.assertTrue(self.cache.has_file(file_name))
        self.assertTrue(self.cache.has_folder(subfolder_name))
        self.assertFalse(self.cache.has_file(sub_file_name))

        self.assertEqual(self.cache.num_files, 1)
        self.assertEqual(subfolder.num_files, 2)
        self.assertEqual(self.cache.num_folders, 1)

        # Another subfolder
        subfolder_2_name = "subfolder_2"
        subfolder_2 = self.cache.make_folder(subfolder_2_name, description="Another test subfolder")
        self.assertEqual(subfolder_2.num_files, 0)
        self.assertEqual(subfolder_2.num_folders, 0)
        self.assertEqual(subfolder.num_folders, 0)
        self.assertEqual(self.cache.num_folders, 2)
        self.assertEqual(subfolder_2.current_folder_name, subfolder_2_name)
        self.assertEqual(subfolder_2.current_folder_description, "Another test subfolder")

        subfolder_2.write_file(
            name=file_name,
            data=sub_data2,
            data_type="npy",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read2_sub_meta2, read2_sub_data2 = subfolder_2.read_file(file_name)
        self.assertTrue(np.allclose(read2_sub_data2, read_sub_data2))
        self.assertEqual(read2_sub_meta2["width"], 100)
        self.assertEqual(read2_sub_meta2["height"], 100)
        self.assertEqual(read2_sub_meta2["channels"], 3)

        # Another level of nesting
        # Another subfolder
        subsubfolder_name = "subfolder_2"
        subsubfolder = subfolder.make_folder(subsubfolder_name, description="A sub-subfolder")
        self.assertEqual(subsubfolder.num_files, 0)
        self.assertEqual(subsubfolder.num_folders, 0)
        self.assertEqual(subfolder.num_folders, 1)
        self.assertEqual(self.cache.num_folders, 2)
        self.assertEqual(subfolder_2.num_folders, 0)
        self.assertEqual(subsubfolder.current_folder_name, subsubfolder_name)
        self.assertEqual(subfolder_2.current_folder_description, "Another test subfolder")

        subsubfolder.write_file(
            name=file_name,
            data=sub_data2,
            data_type="npy",
            metadata={"width": 100, "height": 100, "channels": 3},
        )
        read2_subsub_meta2, read2_subsub_data2 = subsubfolder.read_file(file_name)
        self.assertTrue(np.allclose(read2_sub_data2, read2_subsub_data2))
        self.assertEqual(read2_subsub_meta2["width"], 100)
        self.assertEqual(read2_subsub_meta2["height"], 100)
        self.assertEqual(read2_subsub_meta2["channels"], 3)


def worker_mkfiles(cache_name, cache_description, cache_root, num_files, entry):

    image = np.random.rand(100, 100, 3) * 255
    image = image.astype(np.uint8)

    cache = DatasetCache.get_cache(
        name=cache_name,
        description=cache_description,
        cache_root=cache_root,
    )
    for i in range(num_files):
        file_name = f"{cache_name}xfile_entry{entry}x{i}"
        cache.write_file(
            name=file_name,
            data=image,
            data_type="png",
            metadata={f"width_{entry}_{i}": 100, "height": 100, "channels": 3},
        )
    return


def worker_mkfolders(cache_name, cache_description, cache_root, num_folders, entry):

    image = np.random.rand(100, 100, 3) * 255
    image = image.astype(np.uint8)

    cache = DatasetCache.get_cache(
        name=cache_name,
        description=cache_description,
        cache_root=cache_root,
    )
    for i in range(num_folders):
        file_name = f"{cache_name}xfile_entry{entry}x{i}"
        cache.make_folder(
            name=file_name,
            description=f"Folder for entry {entry} file {i}",
        )
    return


def worker_mkfolders_and_files(cache_name, cache_description, cache_root, num_folders, num_files, entry):

    image = np.random.rand(100, 100, 3) * 255
    image = image.astype(np.uint8)

    cache = DatasetCache.get_cache(
        name=cache_name,
        description=cache_description,
        cache_root=cache_root,
    )
    for i in range(num_folders):
        folder_name = f"{cache_name}xfile_entry{entry}x{i}"
        cache.make_folder(
            name=folder_name,
            description=f"Folder for entry {entry} file {i}",
        )
    for i in range(num_files):
        file_name = f"{cache_name}xfile_entry{entry}x{i}"
        cache.write_file(
            name=file_name,
            data=image,
            data_type="png",
            metadata={f"width_{entry}_{i}": 100, "height": 100, "channels": 3},
        )
    return


class MultiProcessingTest(BasicDatasetCacheTest):
    def test_multiprocessing_create_files(self):
        # This test is a placeholder for multiprocessing tests.
        # In a real scenario, you would implement multiprocessing logic here.
        # For now, we will just check if the cache can be accessed in a separate process.

        with multiprocessing.Pool(processes=4) as pool:
            result = pool.map(
                functools.partial(worker_mkfiles, self.cache_name, self.cache_description, self.cache_root, 5),
                range(10),
            )
        self.assertEqual(self.cache.num_files, 10 * 5)
        for entry in range(10):
            for i in range(5):
                file_name = f"{self.cache_name}xfile_entry{entry}x{i}"
                self.assertTrue(self.cache.has_file(file_name))
                read_meta, read_data = self.cache.read_file(file_name)
                self.assertIsNotNone(read_data)
                self.assertEqual(read_meta[f"width_{entry}_{i}"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)

    def test_multiprocessing_create_folders(self):
        # This test is a placeholder for multiprocessing tests.
        # In a real scenario, you would implement multiprocessing logic here.
        # For now, we will just check if the cache can be accessed in a separate process.

        with multiprocessing.Pool(processes=4) as pool:
            result = pool.map(
                functools.partial(worker_mkfolders, self.cache_name, self.cache_description, self.cache_root, 5),
                range(10),
            )
        self.assertEqual(self.cache.num_folders, 10 * 5)
        for entry in range(10):
            for i in range(5):
                folder_name = f"{self.cache_name}xfile_entry{entry}x{i}"
                self.assertTrue(self.cache.has_folder(folder_name))

    def test_multiprocessing_create_folders_and_files(self):
        # This test is a placeholder for multiprocessing tests.
        # In a real scenario, you would implement multiprocessing logic here.
        # For now, we will just check if the cache can be accessed in a separate process.

        with multiprocessing.Pool(processes=4) as pool:
            result = pool.map(
                functools.partial(
                    worker_mkfolders_and_files, self.cache_name, self.cache_description, self.cache_root, 2, 5
                ),
                range(10),
            )
        self.assertEqual(self.cache.num_folders, 10 * 2)
        for entry in range(10):
            for i in range(2):
                folder_name = f"{self.cache_name}xfile_entry{entry}x{i}"
                self.assertTrue(self.cache.has_folder(folder_name))

        self.assertEqual(self.cache.num_files, 10 * 5)
        for entry in range(10):
            for i in range(5):
                file_name = f"{self.cache_name}xfile_entry{entry}x{i}"
                self.assertTrue(self.cache.has_file(file_name))
                read_meta, read_data = self.cache.read_file(file_name)
                self.assertIsNotNone(read_data)
                self.assertEqual(read_meta[f"width_{entry}_{i}"], 100)
                self.assertEqual(read_meta["height"], 100)
                self.assertEqual(read_meta["channels"], 3)


if __name__ == "__main__":
    unittest.main()
