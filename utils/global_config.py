"""
Global configuration for the Wan model version, sequence length, and frame length.
Values are set exactly once at startup and checked for consistency on subsequent calls.
"""

import torch.distributed as dist
import inspect

WAN_VERSION = None  # None means not yet initialised
SEQ_LEN = None
FRAME_LEN = None


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _caller_func_name() -> str:
    """Walk up the call stack from the caller's caller, skipping anonymous frames
    (<lambda>, <listcomp>, <genexpr>, etc.) until a named function is found."""
    frame = inspect.currentframe().f_back.f_back  # skip this helper itself
    while frame is not None and frame.f_code.co_name.startswith('<'):
        frame = frame.f_back
    return frame.f_code.co_name if frame is not None else '<unknown>'


def get_wan_version(caller: str = "") -> str:
    func_name = _caller_func_name()
    caller_info = f"{caller}.{func_name}" if caller else func_name
    if WAN_VERSION is None:
        raise RuntimeError(
            f"WAN_VERSION has not been initialised. "
            f"Call set_wan_version(args) before using get_wan_version(). "
            f"Called by {caller_info}"
        )
    if _is_rank0():
        print(f"[global_config] get_wan_version='{WAN_VERSION}' by {caller_info}")
    return WAN_VERSION


def get_seq_frame_len(return_seq: bool, return_frame: bool, caller: str = "") -> int:
    func_name = _caller_func_name()
    caller_info = f"{caller}.{func_name}" if caller else func_name
    if SEQ_LEN is None or FRAME_LEN is None:
        raise RuntimeError(
            f"SEQ_LEN/FRAME_LEN have not been initialised. "
            f"Call set_seq_frame_len() before using get_seq_frame_len(). "
            f"Called by {caller_info}"
        )
    if _is_rank0():
        print(f"[global_config] get_seq_frame_len: SEQ_LEN={SEQ_LEN}, FRAME_LEN={FRAME_LEN} by {caller_info}")
    if return_seq:
        return int(SEQ_LEN)
    if return_frame:
        return int(FRAME_LEN)


def _get_arg(args, key: str, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def set_seq_frame_len(args, caller: str = "") -> None:
    global SEQ_LEN, FRAME_LEN

    func_name = _caller_func_name()
    caller_info = f"{caller}.{func_name}" if caller else func_name

    default_shape = [1, 21, 48, 80, 44] if get_wan_version(caller) == "2.2" else [1, 21, 16, 60, 104]
    image_or_video_shape = _get_arg(args, "image_or_video_shape", default_shape)
    frame_len = int((image_or_video_shape[-2] * image_or_video_shape[-1]) / (2 * 2))
    num_frames = int(image_or_video_shape[1])
    slice_last_frames = int(_get_arg(args, "slice_last_frames", min(num_frames, 21)))
    if slice_last_frames <= 0:
        slice_last_frames = min(num_frames, 21)
    seq_frames = min(num_frames, slice_last_frames, 21)
    seq_len = int(seq_frames * frame_len)

    if seq_len in [32760] and frame_len in [1560]:                  # [1, 21, 16, 60, 104]
        assert get_wan_version(caller) == "2.1", f"Expected WAN_VERSION='2.1' for seq_len={seq_len}, frame_len={frame_len}, but got '{get_wan_version(caller)}' by {caller_info}"
    elif seq_len in [8190, 18480] and frame_len in [390, 880]:      # [1, 21, 48, 30, 52], [1, 21, 48, 44, 80]
        assert get_wan_version(caller) == "2.2", f"Expected WAN_VERSION='2.2' for seq_len={seq_len}, frame_len={frame_len}, but got '{get_wan_version(caller)}' by {caller_info}"
    else:
        raise NotImplementedError(
            f"Unexpected seq_len={seq_len} frame_len={frame_len} from image_or_video_shape={image_or_video_shape} "
            f"(num_frames={num_frames}, slice_last_frames={slice_last_frames}, seq_frames={seq_frames})"
        )

    if SEQ_LEN is None and FRAME_LEN is None:
        SEQ_LEN = seq_len
        FRAME_LEN = frame_len
        if _is_rank0():
            print(f"[global_config] SEQ_LEN={SEQ_LEN}, FRAME_LEN={FRAME_LEN} initialised by {caller_info}")
    elif SEQ_LEN == seq_len and FRAME_LEN == frame_len:
        if _is_rank0():
            print(f"[global_config] SEQ_LEN={SEQ_LEN}, FRAME_LEN={FRAME_LEN} consistent by {caller_info}")
    else:
        raise RuntimeError(
            f"SEQ_LEN/FRAME_LEN conflict: global is already set to "
            f"SEQ_LEN={SEQ_LEN}, FRAME_LEN={FRAME_LEN} "
            f"but got SEQ_LEN={seq_len}, FRAME_LEN={frame_len} "
            f"from image_or_video_shape={image_or_video_shape} by {caller_info}."
        )


def set_wan_version(args, caller: str = "") -> str:
    """Detect the wan version from model args and return it.

    First call: sets the global WAN_VERSION and prints the result.
    Subsequent calls: assert the detected version is consistent with the
    already-set value (raise RuntimeError if not) and print the result.

    Args:
        args: model args (dict or namespace) containing model_kwargs.
        caller: str(self.__class__) of the calling class, used for logging.
    """
    global WAN_VERSION

    func_name = _caller_func_name()
    caller_info = f"{caller}.{func_name}" if caller else func_name

    model_kwargs = args.get("model_kwargs", {}) if isinstance(args, dict) else getattr(args, "model_kwargs", {})
    if 'model_name' in model_kwargs and '2.2' in model_kwargs['model_name']:
        detected = "2.2"
    else:
        detected = "2.1"

    if WAN_VERSION is None:
        WAN_VERSION = detected
        if _is_rank0():
            print(f"[global_config] WAN_VERSION initialised to '{WAN_VERSION}' by {caller_info}")
    elif WAN_VERSION == detected:
        if _is_rank0():
            print(f"[global_config] WAN_VERSION consistent: '{WAN_VERSION}' by {caller_info}")
    elif WAN_VERSION != detected:
        raise RuntimeError(
            f"WAN_VERSION conflict: global is already set to '{WAN_VERSION}' "
            f"but detected '{detected}' from model_name "
            f"'{model_kwargs.get('model_name', '<unknown>')}' by {caller_info}. "
            "All components must use the same Wan version."
        )

    set_seq_frame_len(args, caller)

    return WAN_VERSION
