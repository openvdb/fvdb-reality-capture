import pathlib

import cv2

# import matplotlib.pyplot as plt
import numpy as np
import pye57
import tqdm

from ..sfm_scene import (
    SfmCache,
    SfmCameraMetadata,
    SfmCameraType,
    SfmImageMetadata,
    SfmScene,
)


def normalized_quat_to_rotmat(quat_: np.ndarray) -> np.ndarray:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat_.shape[-1] == 4, quat_.shape

    w, x, y, z = np.split(quat_, 4, axis=-1)
    w, x, y, z = w.squeeze(-1), x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)

    mat = np.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        axis=-1,
    )
    return mat.reshape(quat_.shape[:-1] + (3, 3))

class E57Scan:
    """
    Class representing a single scan from an E57 file.
    """

    def __init__(self, e57: pye57.E57, index: int):
        self.e57 = e57
        self.index = index

        self._scan_read = False

    def _get_data(self):
        if not self._scan_read:
            self.scan_data = self.e57.read_scan(self.index, colors=True)
            self._scan_read = True
        return self.scan_data


    def get_points(self) -> np.ndarray:
        data = self._get_data()
        return np.column_stack(
            [
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"],
            ]
        )

    def get_colors(self) -> np.ndarray:
        data = self._get_data()
        return np.column_stack(
            [
                data["colorRed"],
                data["colorGreen"],
                data["colorBlue"],
            ]
        )

    def get_rotation(self):
        return self.e57.scan_position(self.index)


class E57File:
    """
    Class representing a single E57 file which contains a set of scans and a set of images and camera metadata.
    """

    def __init__(self, file_path: str | pathlib.Path):
        self.file_path = pathlib.Path(file_path)
        self.e57 = pye57.E57(str(file_path))

    def __len__(self):
        return self.e57.scan_count

    def _get_header(self, index: int) -> pye57.ScanHeader:
        return self.e57.get_header(index)

    def get_scan(self, index: int) -> E57Scan:
        return E57Scan(self.e57, index)

    def get_num_images(self) -> int:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        return len(images)

    def get_image(self, index: int) -> np.ndarray:  # [H, W, 3]
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]

        if image_node.isDefined("pinholeRepresentation"):
            representation = image_node["pinholeRepresentation"]
        elif image_node.isDefined("sphericalRepresentation"):
            representation = image_node["sphericalRepresentation"]
        else:
            raise ValueError("No image representation found")
        jpeg_image = representation["jpegImage"]
        jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)
        jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
        image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)

        assert image is not None

        return image

    def get_intrinsics(self, index: int) -> tuple[int, int, float, float, float, float]:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]

        if image_node.isDefined("pinholeRepresentation"):
            representation = image_node["pinholeRepresentation"]

            # Get focal length (in meters) and pixel dimensions (in meters)
            focal_length_m = representation["focalLength"].value()
            pixel_width_m = representation["pixelWidth"].value()
            pixel_height_m = representation["pixelHeight"].value()
            principal_point = representation["principalPointX"].value(), representation["principalPointY"].value()

            # Convert focal length from meters to pixels
            fx = focal_length_m / pixel_width_m
            fy = focal_length_m / pixel_height_m

            nx = representation["imageWidth"].value()
            ny = representation["imageHeight"].value()

            return nx, ny, fx, fy, principal_point[0], principal_point[1]

        elif image_node.isDefined("sphericalRepresentation"):
            raise ValueError("Spherical representation not supported")
        else:
            raise ValueError("No camera representation found")

    def get_camera_to_world_matrix(self, index: int) -> np.ndarray:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]
        pose_node = image_node["pose"]
        rot_node = pose_node["rotation"]
        trans_node = pose_node["translation"]

        rot = np.array([rot_node["w"].value(), rot_node["x"].value(), rot_node["y"].value(), rot_node["z"].value()])
        trans = np.array([trans_node["x"].value(), trans_node["y"].value(), trans_node["z"].value()])

        rot_mat = normalized_quat_to_rotmat(rot)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_mat
        transform_matrix[:3, 3] = trans

        # The E57 standard specifies a Z-up camera coordinate system whereas our colmap code expects Y-up.
        # This is a 180-degree rotation around the X-axis, which flips the Y and Z axes.
        cv_to_e57_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        transform_matrix = transform_matrix @ cv_to_e57_transform

        return transform_matrix

def load_scan_for_nksr(file_path: pathlib.Path, downsample_point_factor: int = 10):
    e57_file = E57File(file_path)
    scan = e57_file.get_scan(0)
    points = scan.get_points()
    points_rgb = scan.get_colors()
    location = scan.get_rotation()

    points = points[::downsample_point_factor]
    points_rgb = points_rgb[::downsample_point_factor]

    return points, points_rgb, location


def load_single_e57(file_path: pathlib.Path,
                    cache: SfmCache,
                    cumulative_num_points: int,
                    camera_metadata: dict[int,SfmCameraMetadata],
                    image_metadata: list[SfmImageMetadata],
                    downsample_point_factor: int = 10):

    e57_file = E57File(file_path)

    for i in range(len(e57_file)):
        scan = e57_file.get_scan(i)
        points = scan.get_points()
        points_rgb = scan.get_colors()

    assert len(e57_file) == 1

    points = points[::downsample_point_factor]
    points_rgb = points_rgb[::downsample_point_factor]
    # points_err = np.zeros(points.shape[0],dtype=points.dtype)


    cumulative_num_cameras = len(camera_metadata)
    cumulative_num_images = len(image_metadata)
    for i in range(e57_file.get_num_images()):
        cam_id = cumulative_num_cameras + i
        image_id = cumulative_num_images + i

        image = e57_file.get_image(i)
        cache_image_metadata = cache.write_file(
            f"image_{image_id:04}",
            image,
            "jpg",
            quality=98
        )

        nx, ny, fx, fy, cx, cy = e57_file.get_intrinsics(i)
        camera_metadata[cam_id] = SfmCameraMetadata(
            nx, ny, fx, fy, cx, cy, SfmCameraType.PINHOLE,np.array([])
        )

        projection_matrix = camera_metadata[cam_id].projection_matrix
        camera_to_world_matrix = e57_file.get_camera_to_world_matrix(i)
        world_to_camera_matrix = np.linalg.inv(camera_to_world_matrix)

        points_cam = world_to_camera_matrix[:3,:3] @ points.T + world_to_camera_matrix[:3,3][:,None]
        points_clip = projection_matrix @ points_cam
        points_pix = (points_clip[:2] / points_clip[-1]).T
        mask = np.logical_and.reduce([
            points_pix[:,0] >= 0,
            points_pix[:,1] >= 0,
            points_pix[:,0] < nx,
            points_pix[:,1] < ny,
        ])
        points_in_image = np.where(mask)[0] + cumulative_num_points

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # plt.subplot(1,2,2)
        # plt.scatter(
        #     points_pix[mask,0],
        #     points_pix[mask,1],
        #     c=colors[mask].astype(np.float32)/255.0
        # )
        # plt.show()


        image_metadata.append(SfmImageMetadata(
            camera_to_world_matrix=camera_to_world_matrix,
            world_to_camera_matrix=world_to_camera_matrix,
            camera_metadata=camera_metadata[cam_id],
            camera_id=cam_id,
            image_path=str(cache_image_metadata["path"]),
            mask_path="",
            point_indices=points_in_image,
            image_id=image_id
        ))

    return camera_metadata, image_metadata, points, points_rgb

def load_e57_scene(data_path: pathlib.Path, downsample_point_factor: int = 10 ):

    all_files = data_path.glob("*.e57")
    camera_metadata: dict[int,SfmCameraMetadata] = {}
    image_metadata: list[SfmImageMetadata] = []
    cache = SfmCache.get_cache(
        data_path/'_cache',
        "e57_dataset_cache",
        "cache for e57 data"
    )

    points = []
    points_rgb = []
    cumulative_num_points = 0


    for file_path in tqdm.tqdm(all_files,desc="loading e57 files"):
        camera_metadata, image_metadata, points_i, points_rgb_i = load_single_e57(
            file_path=file_path,
            cache=cache,
            cumulative_num_points=cumulative_num_points,
            camera_metadata=camera_metadata,
            image_metadata=image_metadata,
            downsample_point_factor=downsample_point_factor)

        cumulative_num_points += points_i.shape[0]
        points.append(points_i)
        points_rgb.append(points_rgb_i)

    points_cat = np.concatenate(points, axis=0)
    points_rgb_cat = np.concatenate(points_rgb, axis=0)
    points_err = np.zeros(points_cat.shape[0], dtype=points_cat.dtype)

    return SfmScene(camera_metadata,image_metadata,points_cat,points_err,points_rgb_cat,None,None,cache)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger().name = "e57_dataset"
    # file_path = "/home/bbartlett/Data1/nuRec/hexagon/20042020-kantonalschule-_Setip_001.e57"

    # load_single_e57(pathlib.Path(file_path))    # logging.getLogger().name = "e57_dataset"
    pass

