from importlib import metadata
import os
import sys
import platform
import subprocess
import time
import logging
import warnings
import importlib

import hashlib
import json
import yaml
import requests
import torch
import onnxruntime as ort
from tqdm import tqdm


class Separator:
    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="/tmp/audio-separator-models/",
        output_dir=None,
        output_format="WAV",
        normalization_threshold=0.9,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": False,
        },
        vr_params={
            "batch_size": 2, # decrease to 2 from 16 due to GPU RAM
            "window_size": 512,
            "aggression": 5,
            "enable_tta": False,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        },
        demucs_params={
            "segment_size": "Default",
            "shifts": 2,
            "overlap": 0.25,
            "segments_enabled": True,
        },
        mdxc_params={"segment_size": 256, "batch_size": 1, "overlap": 8},
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        self.model_file_dir = model_file_dir
        self.output_dir = output_dir

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)

        self.output_format = output_format

        if self.output_format is None:
            self.output_format = "WAV"

        self.normalization_threshold = normalization_threshold

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            self.logger.debug(
                f"Single stem output requested, only one output file ({output_single_stem}) will be written"
            )

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(
                f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality, but is slightly slower."
            )

        self.sample_rate = sample_rate

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.arch_specific_params = {
            "MDX": mdx_params,
            "VR": vr_params,
            "Demucs": demucs_params,
            "MDXC": mdxc_params,
        }

        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None

        self.onnx_execution_provider = None
        self.model_instance = None

        self.model_is_uvr_vip = False
        self.model_friendly_name = None

        self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        self.log_onnxruntime_packages()
        self.setup_torch_device()

    def log_onnxruntime_packages(self):
        """
        This method logs the ONNX Runtime package versions, including the GPU and Silicon packages if available.
        """
        onnxruntime_gpu_package = self.get_package_distribution("onnxruntime-gpu")
        onnxruntime_silicon_package = self.get_package_distribution(
            "onnxruntime-silicon"
        )
        onnxruntime_cpu_package = self.get_package_distribution("onnxruntime")

    def setup_torch_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
        """
        hardware_acceleration_enabled = False
        ort_providers = ort.get_available_providers()

        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info(
                "No hardware acceleration could be configured, running in CPU mode"
            )
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        """
        This method configures the CUDA device for PyTorch and ONNX Runtime, if available.
        """
        self.torch_device = torch.device("cuda")
        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info(
                "ONNXruntime has CUDAExecutionProvider available, enabling acceleration"
            )
            self.onnx_execution_provider = ["CUDAExecutionProvider"]

    def configure_mps(self, ort_providers):
        """
        This method configures the Apple Silicon MPS/CoreML device for PyTorch and ONNX Runtime, if available.
        """
        self.logger.info(
            "Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS"
        )
        self.torch_device_mps = torch.device("mps")

        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info(
                "ONNXruntime has CoreMLExecutionProvider available, enabling acceleration"
            )
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else:
            self.logger.warning(
                "CoreMLExecutionProvider not available in ONNXruntime, so acceleration will NOT be enabled"
            )

    def get_package_distribution(self, package_name):
        """
        This method returns the package distribution for a given package name if installed, or None otherwise.
        """
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Python package: {package_name} not installed")
            return None

    def get_model_hash(self, model_path):
        """
        This method returns the MD5 hash of a given model file.
        """
        self.logger.debug(f"Calculating hash of model file {model_path}")
        try:
            # Open the model file in binary read mode
            with open(model_path, "rb") as f:
                # Move the file pointer 10MB before the end of the file
                f.seek(-10000 * 1024, 2)
                # Read the file from the current pointer to the end and calculate its MD5 hash
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            # If an IOError occurs (e.g., if the file is less than 10MB large), log the error
            self.logger.error(
                f"IOError seeking -10MB or reading model file for hash calculation: {e}"
            )
            # Attempt to open the file again, read its entire content, and calculate the MD5 hash
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file_if_not_exists(self, url, output_path):
        """
        This method downloads a file from a given URL to a given output path, if the file does not already exist.
        """

        if os.path.isfile(output_path):
            self.logger.debug(
                f"File already exists at {output_path}, skipping download"
            )
            return

        self.logger.debug(
            f"Downloading file from {url} to {output_path} with timeout 300s"
        )
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        else:
            raise RuntimeError(
                f"Failed to download file from {url}, response code: {response.status_code}"
            )

    def list_supported_model_files(self):
        """
        This method lists the supported model files for audio-separator, by fetching the same file UVR uses to list these.
        """
        download_checks_path = os.path.join(self.model_file_dir, "download_checks.json")

        self.download_file_if_not_exists(
            "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json",
            download_checks_path,
        )

        model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))
        self.logger.debug(f"Model download list loaded")

        # model_downloads_list JSON structure / example snippet:
        # {
        #     "vr_download_list": {
        #             "VR Arch Single Model v5: 1_HP-UVR": "1_HP-UVR.pth",
        #             "VR Arch Single Model v5: UVR-DeNoise by FoxJoy": "UVR-DeNoise.pth",
        #     },
        #     "mdx_download_list": {
        #             "MDX-Net Model: UVR-MDX-NET Inst HQ 3": "UVR-MDX-NET-Inst_HQ_3.onnx",
        #             "MDX-Net Model: UVR-MDX-NET Karaoke 2": "UVR_MDXNET_KARA_2.onnx",
        #             "MDX-Net Model: Kim Vocal 2": "Kim_Vocal_2.onnx",
        #             "MDX-Net Model: kuielab_b_drums": "kuielab_b_drums.onnx"
        #     },
        #     "demucs_download_list": {
        #             "Demucs v4: htdemucs_ft": {
        #                     "f7e0c4bc-ba3fe64a.th": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
        #                     "d12395a8-e57c48e6.th": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
        #                     "92cfc3b6-ef3bcb9c.th": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
        #                     "04573f0d-f3cf25b2.th": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
        #                     "htdemucs_ft.yaml": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_ft.yaml"
        #             },
        #             "Demucs v4: htdemucs": {
        #                     "955717e8-8726e21a.th": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
        #                     "htdemucs.yaml": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs.yaml"
        #             },
        #             "Demucs v1: tasnet": {
        #                     "tasnet.th": "https://dl.fbaipublicfiles.com/demucs/v2.0/tasnet.th"
        #             },
        #     },
        #     "mdx23_download_list": {
        #             "MDX23C Model: MDX23C_D1581": {
        #                     "MDX23C_D1581.ckpt": "model_2_stem_061321.yaml"
        #             }
        #     },
        #     "mdx23c_download_list": {
        #             "MDX23C Model: MDX23C-InstVoc HQ": {
        #                     "MDX23C-8KFFT-InstVoc_HQ.ckpt": "model_2_stem_full_band_8k.yaml"
        #             }
        #     },
        #     "roformer_download_list": {
        #             "Roformer Model: BS-Roformer-Viperx-1297": {
        #                     "model_bs_roformer_ep_317_sdr_12.9755.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
        #             },
        #             "Roformer Model: BS-Roformer-Viperx-1296": {
        #                     "model_bs_roformer_ep_368_sdr_12.9628.ckpt": "model_bs_roformer_ep_368_sdr_12.9628.yaml"
        #             },
        #             "Roformer Model: BS-Roformer-Viperx-1053": {
        #                     "model_bs_roformer_ep_937_sdr_10.5309.ckpt": "model_bs_roformer_ep_937_sdr_10.5309.yaml"
        #             },
        #             "Roformer Model: Mel-Roformer-Viperx-1143": {
        #                     "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
        #             }
        #     },
        # }

        # Only show Demucs v4 models as we've only implemented support for v4
        filtered_demucs_v4 = {
            key: value
            for key, value in model_downloads_list["demucs_download_list"].items()
            if key.startswith("Demucs v4")
        }

        # Return object with list of model names, which are the keys in vr_download_list, mdx_download_list, demucs_download_list, mdx23_download_list, mdx23c_download_list, grouped by type: VR, MDX, Demucs, MDX23, MDX23C
        model_files_grouped_by_type = {
            "VR": model_downloads_list["vr_download_list"],
            "MDX": {
                **model_downloads_list["mdx_download_list"],
                **model_downloads_list["mdx_download_vip_list"],
            },
            "Demucs": filtered_demucs_v4,
            "MDXC": {
                **model_downloads_list["mdx23c_download_list"],
                **model_downloads_list["mdx23c_download_vip_list"],
                **model_downloads_list["roformer_download_list"],
            },
        }
        return model_files_grouped_by_type

    def print_uvr_vip_message(self):
        """
        This method prints a message to the user if they have downloaded a VIP model, reminding them to support Anjok07 on Patreon.
        """
        if self.model_is_uvr_vip:
            self.logger.warning(
                f"The model: '{self.model_friendly_name}' is a VIP model, intended by Anjok07 for access by paying subscribers only."
            )
            self.logger.warning(
                "If you are not already subscribed, please consider supporting the developer of UVR, Anjok07 by subscribing here: https://patreon.com/uvr"
            )

    def download_model_files(self, model_filename):
        """
        This method downloads the model files for a given model filename, if they are not already present.
        """
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")

        supported_model_files_grouped = self.list_supported_model_files()
        public_model_repo_url_prefix = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
        vip_model_repo_url_prefix = (
            "https://github.com/Anjok0109/ai_magic/releases/download/v5"
        )

        yaml_config_filename = None

        self.logger.debug(
            f"Searching for model_filename {model_filename} in supported_model_files_grouped"
        )
        for model_type, model_list in supported_model_files_grouped.items():
            for model_friendly_name, model_download_list in model_list.items():
                self.model_is_uvr_vip = "VIP" in model_friendly_name
                model_repo_url_prefix = (
                    vip_model_repo_url_prefix
                    if self.model_is_uvr_vip
                    else public_model_repo_url_prefix
                )

                # If model_download_list is a string, this model only requires a single file so we can just download it
                if (
                    isinstance(model_download_list, str)
                    and model_download_list == model_filename
                ):
                    self.logger.debug(
                        f"Single file model identified: {model_friendly_name}"
                    )
                    self.model_friendly_name = model_friendly_name

                    self.download_file_if_not_exists(
                        f"{model_repo_url_prefix}/{model_filename}", model_path
                    )
                    self.print_uvr_vip_message()

                    self.logger.debug(
                        f"Returning path for single model file: {model_path}"
                    )
                    return (
                        model_filename,
                        model_type,
                        model_friendly_name,
                        model_path,
                        yaml_config_filename,
                    )

                # If it's a dict, iterate through each entry check if any of them match model_filename
                # If the value is a full URL, download it from that URL.
                # If it's just a filename, add the model repo prefix to get the URL to download.
                elif isinstance(model_download_list, dict):
                    this_model_matches_input_filename = False
                    for file_name, file_url in model_download_list.items():
                        if file_name == model_filename or file_url == model_filename:
                            self.logger.debug(
                                f"Found input filename {model_filename} in multi-file model: {model_friendly_name}"
                            )
                            this_model_matches_input_filename = True

                    if this_model_matches_input_filename:
                        self.logger.debug(
                            f"Multi-file model identified: {model_friendly_name}, iterating through files to download"
                        )
                        self.model_friendly_name = model_friendly_name
                        self.print_uvr_vip_message()

                        for config_key, config_value in model_download_list.items():
                            self.logger.debug(
                                f"Attempting to identify download URL for config pair: {config_key} -> {config_value}"
                            )

                            # Demucs models have full URLs to download from Facebook repos, and config_key is set to the file name
                            if config_value.startswith("http"):
                                self.download_file_if_not_exists(
                                    config_value,
                                    os.path.join(self.model_file_dir, config_key),
                                )

                            # Checkpoint models apparently use config_key as the model filename, but the value is a YAML config file name...
                            # Both need to be downloaded, but the model data YAML file actually comes from the application data repo...
                            elif config_key.endswith(".ckpt"):
                                download_url = f"{model_repo_url_prefix}/{config_key}"
                                self.download_file_if_not_exists(
                                    download_url,
                                    os.path.join(self.model_file_dir, config_key),
                                )

                                # In case the user specified the YAML filename as the model input instead of the model filename, correct that
                                if model_filename.endswith(".yaml"):
                                    self.logger.warning(
                                        f"The model name you've specified, {model_filename} is actually a model config file, not a model file itself."
                                    )
                                    self.logger.warning(
                                        f"We found a model matching this config file: {config_key} so we'll use that model file for this run."
                                    )
                                    self.logger.warning(
                                        "To prevent confusing / inconsistent behaviour in future, specify an actual model filename instead."
                                    )
                                    model_filename = config_key
                                    model_path = os.path.join(
                                        self.model_file_dir, f"{model_filename}"
                                    )

                                # For MDXC models, the config_value is the YAML file which needs to be downloaded separately from the application_data repo
                                yaml_config_filename = config_value
                                yaml_config_filepath = os.path.join(
                                    self.model_file_dir, yaml_config_filename
                                )

                                # Repo for model data and configuration sources from UVR
                                model_data_url_prefix = "https://raw.githubusercontent.com/TRvlvr/application_data/main"
                                yaml_config_url = f"{model_data_url_prefix}/mdx_model_data/mdx_c_configs/{yaml_config_filename}"

                                self.download_file_if_not_exists(
                                    f"{yaml_config_url}", yaml_config_filepath
                                )

                            # MDX and VR models have config_value set to the model filename
                            else:
                                download_url = f"{model_repo_url_prefix}/{config_value}"
                                self.download_file_if_not_exists(
                                    download_url,
                                    os.path.join(self.model_file_dir, config_value),
                                )

                        self.logger.debug(
                            f"All files downloaded for model {model_friendly_name}, returning initial path {model_path}"
                        )
                        return (
                            model_filename,
                            model_type,
                            model_friendly_name,
                            model_path,
                            yaml_config_filename,
                        )

        raise ValueError(
            f"Model file {model_filename} not found in supported model files"
        )

    def load_model_data_from_yaml(self, yaml_config_filename):
        """
        This method loads model-specific parameters from the YAML file for that model.
        The parameters in the YAML are critical to inferencing, as they need to match whatever was used during training.
        """
        # Verify if the YAML filename includes a full path or just the filename
        if not os.path.exists(yaml_config_filename):
            model_data_yaml_filepath = os.path.join(
                self.model_file_dir, yaml_config_filename
            )
        else:
            model_data_yaml_filepath = yaml_config_filename

        self.logger.debug(
            f"Loading model data from YAML at path {model_data_yaml_filepath}"
        )

        model_data = yaml.load(
            open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader
        )
        self.logger.debug(f"Model data loaded from YAML file: {model_data}")

        if "roformer" in model_data_yaml_filepath:
            model_data["is_roformer"] = True

        return model_data

    def load_model_data_using_hash(self, model_path):
        """
        This method loads model-specific parameters from UVR model data files.
        These parameters are critical to inferencing using a given model, as they need to match whatever was used during training.
        The correct parameters are identified by calculating the hash of the model file and looking up the hash in the UVR data files.
        """
        # Model data and configuration sources from UVR
        model_data_url_prefix = (
            "https://raw.githubusercontent.com/TRvlvr/application_data/main"
        )

        vr_model_data_url = f"{model_data_url_prefix}/vr_model_data/model_data_new.json"
        mdx_model_data_url = (
            f"{model_data_url_prefix}/mdx_model_data/model_data_new.json"
        )

        # Calculate hash for the downloaded model
        self.logger.debug(
            "Calculating MD5 hash for model file to identify model parameters from UVR data..."
        )
        model_hash = self.get_model_hash(model_path)
        self.logger.debug(f"Model {model_path} has hash {model_hash}")

        # Setting up the path for model data and checking its existence
        vr_model_data_path = os.path.join(self.model_file_dir, "vr_model_data.json")
        self.logger.debug(f"VR model data path set to {vr_model_data_path}")
        self.download_file_if_not_exists(vr_model_data_url, vr_model_data_path)

        mdx_model_data_path = os.path.join(self.model_file_dir, "mdx_model_data.json")
        self.logger.debug(f"MDX model data path set to {mdx_model_data_path}")
        self.download_file_if_not_exists(mdx_model_data_url, mdx_model_data_path)

        # Loading model data
        self.logger.debug(
            "Loading MDX and VR model parameters from UVR model data files..."
        )
        vr_model_data_object = json.load(open(vr_model_data_path, encoding="utf-8"))
        mdx_model_data_object = json.load(open(mdx_model_data_path, encoding="utf-8"))

        # vr_model_data_object JSON structure / example snippet:
        # {
        #     "0d0e6d143046b0eecc41a22e60224582": {
        #         "vr_model_param": "3band_44100_mid",
        #         "primary_stem": "Instrumental"
        #     },
        #     "6b5916069a49be3fe29d4397ecfd73fa": {
        #         "vr_model_param": "3band_44100_msb2",
        #         "primary_stem": "Instrumental",
        #         "is_karaoke": true
        #     },
        #     "0ec76fd9e65f81d8b4fbd13af4826ed8": {
        #         "vr_model_param": "4band_v3",
        #         "primary_stem": "No Woodwinds"
        #     },
        #     "0fb9249ffe4ffc38d7b16243f394c0ff": {
        #         "vr_model_param": "4band_v3",
        #         "primary_stem": "No Reverb"
        #     },
        #     "6857b2972e1754913aad0c9a1678c753": {
        #         "vr_model_param": "4band_v3",
        #         "primary_stem": "No Echo",
        #         "nout": 48,
        #         "nout_lstm": 128
        #     },
        #     "944950a9c5963a5eb70b445d67b7068a": {
        #         "vr_model_param": "4band_v3_sn",
        #         "primary_stem": "Vocals",
        #         "nout": 64,
        #         "nout_lstm": 128,
        #         "is_karaoke": false,
        #         "is_bv_model": true,
        #         "is_bv_model_rebalanced": 0.9
        #     }
        # }

        # mdx_model_data_object JSON structure / example snippet:
        # {
        #     "0ddfc0eb5792638ad5dc27850236c246": {
        #         "compensate": 1.035,
        #         "mdx_dim_f_set": 2048,
        #         "mdx_dim_t_set": 8,
        #         "mdx_n_fft_scale_set": 6144,
        #         "primary_stem": "Vocals"
        #     },
        #     "26d308f91f3423a67dc69a6d12a8793d": {
        #         "compensate": 1.035,
        #         "mdx_dim_f_set": 2048,
        #         "mdx_dim_t_set": 9,
        #         "mdx_n_fft_scale_set": 8192,
        #         "primary_stem": "Other"
        #     },
        #     "2cdd429caac38f0194b133884160f2c6": {
        #         "compensate": 1.045,
        #         "mdx_dim_f_set": 3072,
        #         "mdx_dim_t_set": 8,
        #         "mdx_n_fft_scale_set": 7680,
        #         "primary_stem": "Instrumental"
        #     },
        #     "2f5501189a2f6db6349916fabe8c90de": {
        #         "compensate": 1.035,
        #         "mdx_dim_f_set": 2048,
        #         "mdx_dim_t_set": 8,
        #         "mdx_n_fft_scale_set": 6144,
        #         "primary_stem": "Vocals",
        #         "is_karaoke": true
        #     },
        #     "2154254ee89b2945b97a7efed6e88820": {
        #         "config_yaml": "model_2_stem_061321.yaml"
        #     },
        #     "116f6f9dabb907b53d847ed9f7a9475f": {
        #         "config_yaml": "model_2_stem_full_band_8k.yaml"
        #     }
        # }

        if model_hash in mdx_model_data_object:
            model_data = mdx_model_data_object[model_hash]
        elif model_hash in vr_model_data_object:
            model_data = vr_model_data_object[model_hash]
        else:
            raise ValueError(
                f"Unsupported Model File: parameters for MD5 hash {model_hash} could not be found in UVR model data file for MDX or VR arch."
            )

        self.logger.debug(
            f"Model data loaded from UVR JSON using hash {model_hash}: {model_data}"
        )

        return model_data

    def load_model(
        self, model_filename="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    ):
        """
        This method instantiates the architecture-specific separation class,
        loading the separation model into memory, downloading it first if necessary.
        """
        self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()

        # Setting up the model path
        (
            model_filename,
            model_type,
            model_friendly_name,
            model_path,
            yaml_config_filename,
        ) = self.download_model_files(model_filename)
        model_name = model_filename.split(".")[0]
        self.logger.debug(
            f"Model downloaded, friendly name: {model_friendly_name}, model_path: {model_path}"
        )

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "onnx_execution_provider": self.onnx_execution_provider,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
        }

        # Instantiate the appropriate separator class depending on the model type
        separator_classes = {
            "MDX": "mdx_separator.MDXSeparator",
            "VR": "vr_separator.VRSeparator",
            "Demucs": "demucs_separator.DemucsSeparator",
            "MDXC": "mdxc_separator.MDXCSeparator",
        }

        if (
            model_type not in self.arch_specific_params
            or model_type not in separator_classes
        ):
            raise ValueError(f"Model type not supported (yet): {model_type}")

        if model_type == "Demucs" and sys.version_info < (3, 10):
            raise Exception("Demucs models require Python version 3.10 or newer.")

        self.logger.debug(
            f"Importing module for model type {model_type}: {separator_classes[model_type]}"
        )

        module_name, class_name = separator_classes[model_type].split(".")
        module = importlib.import_module(f"uvr.architectures.{module_name}")
        separator_class = getattr(module, class_name)

        self.logger.debug(
            f"Instantiating separator class for model type {model_type}: {separator_class}"
        )
        self.model_instance = separator_class(
            common_config=common_params,
            arch_config=self.arch_specific_params[model_type],
        )

        # Log the completion of the model load process
        self.logger.debug("Loading model completed.")
        self.logger.info(
            f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}'
        )

    def separate(self, audio_file_path):
        """
        Separates the audio file into different stems (e.g., vocals, instruments) using the loaded model.

        This method takes the path to an audio file, processes it through the loaded separation model, and returns
        the paths to the output files containing the separated audio stems. It handles the entire flow from loading
        the audio, running the separation, clearing up resources, and logging the process.

        Parameters:
        - audio_file_path (str): The path to the audio file to be separated.

        Returns:
        - output_files (list of str): A list containing the paths to the separated audio stem files.
        """
        # Starting the separation process
        self.logger.info(
            f"Starting separation process for audio_file_path: {audio_file_path}"
        )
        separate_start_time = time.perf_counter()

        self.logger.debug(
            f"Normalization threshold set to {self.normalization_threshold}, waveform will lowered to this max amplitude to avoid clipping."
        )

        # Run separation method for the loaded model
        output_files = self.model_instance.separate(audio_file_path)

        # Clear GPU cache to free up memory
        self.model_instance.clear_gpu_cache()

        # Unset more separation params to prevent accidentally re-using the wrong source files or output paths
        self.model_instance.clear_file_specific_paths()

        # Remind the user one more time if they used a VIP model, so the message doesn't get lost in the logs
        self.print_uvr_vip_message()

        # Log the completion of the separation process
        self.logger.debug("Separation process completed.")
        self.logger.info(
            f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}'
        )

        return output_files
