import os

from torch.utils.data import DataLoader
from rawdiffusion.datasets.transforms import ImageTransforms
from .raw_image_dataset import RAWImageDataset
from rawdiffusion.datasets.camera import get_camera


def create_dataset(
    *,
    camera_name,
    data_dir,
    file_list,
    batch_size,
    seed,
    is_train=True,
    transform=True,
    permutate_once=False,
    resample_dataset_size=None,
    min_mode="black_level",
    patch_size=256,
    max_items=None,
    num_workers=8,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    camera_names = camera_name.split(",")
    file_lists = file_list.split(",")

    # currently only support single dataset with multiple models
    dataset_folder_name = os.path.basename(os.path.normpath(data_dir)).lower()
    dataset_name = dataset_folder_name.split("_")[0]

    if transform:
        transforms = ImageTransforms(
            patch_size=patch_size,
            is_train=is_train,
        )
    else:
        transforms = None

    assert min_mode in ["min_value", "black_level"]

    raw_min_values = []
    raw_max_values = []

    for camera_name in camera_names:
        if dataset_name in ["fivek", "nod"]:
            camera = get_camera(dataset_name, camera_name)
            if camera is None:
                raise ValueError(f"unknown camera_name: {camera_name}")
            if min_mode == "min_value":
                raw_min_values.append(camera.min_value)
            elif min_mode == "black_level":
                raw_min_values.append(camera.black_level)
            else:
                raise ValueError(f"unknown min_mode: {min_mode}")

            raw_max_values.append(camera.white_level)
        elif dataset_name in ["cityscapes", "bdd100k", "pascalraw"]:
            raw_min_values.append(None)
            raw_max_values.append(None)
        else:
            raise ValueError(f"unknown dataset_name: {dataset_name}")

    if max_items is not None:
        new_file_lists = []
        for file_list in file_lists:
            name, ext = os.path.splitext(file_list)
            # file_list = f"{name}_{max_items}_{seed}{ext}"
            file_list = f"{name}{ext}"
            new_file_lists.append(file_list)
        file_lists = new_file_lists

    dataset = RAWImageDataset(
        file_lists=file_lists,
        dataset_path=data_dir,
        raw_min_values=raw_min_values,
        raw_max_values=raw_max_values,
        transforms=transforms,
    )

    if permutate_once:
        from .dataset_wrapper import PermutedDataset

        dataset = PermutedDataset(dataset, seed=123)

    if resample_dataset_size is not None:
        from .dataset_wrapper import RandomSampleDataset

        dataset = RandomSampleDataset(dataset, n=resample_dataset_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )
    return loader
