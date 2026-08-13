import os
import io
import gc
import json
import base64
import random
import torch
import logging

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *

import folder_paths
import comfy.model_management as mm
import comfy.utils

from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except ImportError:
    MTMDChatHandler = None
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import (
        ChatFormatterResponse,
        chat_formatter_to_chat_completion_handler,
    )
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    _TEXT_TEMPLATE = True
except ImportError:
    _TEXT_TEMPLATE = False

chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except ImportError:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except ImportError:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL"]
except ImportError:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except ImportError:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]
except ImportError:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except ImportError:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except ImportError:
    LFM25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except ImportError:
    GraniteDoclingChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

def _text_only_chat_handler(handler_cls, llama, enable_thinking=False, preserve_thinking=False):
    """Build a non-MTMD chat handler from an MTMD handler's Jinja template."""
    if not _TEXT_TEMPLATE:
        return None
    eos = llama.detokenize([llama.token_eos()]).decode("utf-8", errors="ignore")
    bos = llama.detokenize([llama.token_bos()]).decode("utf-8", errors="ignore")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        handler_cls.CHAT_FORMAT
    )

    class _Formatter:
        def __call__(
            self,
            *,
            messages,
            functions=None,
            function_call=None,
            tools=None,
            tool_choice=None,
            **kwargs,
        ):
            def raise_exception(message):
                raise ValueError(message)

            prompt = env.render(
                messages=messages,
                eos_token=eos,
                bos_token=bos,
                raise_exception=raise_exception,
                add_generation_prompt=True,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
                add_vision_id=True,
            )
            return ChatFormatterResponse(prompt=prompt, stop=[eos], added_special=True)

    return chat_formatter_to_chat_completion_handler(_Formatter())

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    #states = {}
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            #cls.states.clear()
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            #cls.states.pop(f"{id}", None)
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            cls.llm.close()
        except Exception:
            pass
            
        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen2.5-VL":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv26ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')
        
        cls.clean(all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        n_gpu_layers = -1
        
        model_path = resolve_llm_gguf(model, "Model")
        handler = get_chat_handler(chat_handler)
        
        if vram_limit == 0:
            n_gpu_layers = 0
        elif vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path)  * 1.55 / (1024 ** 3)
            gguf_layer_size = max(gguf_size / gguf_layers, 1e-6)
        
        if mmproj and mmproj != "None":
            mmproj_path = resolve_llm_gguf(mmproj, "mmproj")
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None!')
            
            if vram_limit > 0:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip:  {mmproj}")
            
            think_mode = "Thinking" in chat_handler
            # 0.3.46+ uses mmproj_path; older builds use clip_model_path.
            kwargs = {
                "verbose": False,
                "use_gpu": n_gpu_layers != 0,
            }
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
            elif any(h in chat_handler for h in ("MiniCPM-v4.5", "GLM-4.6V", "Qwen3.5", "Qwen3.6")):
                kwargs["enable_thinking"] = think_mode
            # MTMD only applies overrides when > 0; pass for kwargs-only subclasses too.
            if image_max_tokens > 0:
                kwargs["image_max_tokens"] = image_max_tokens
            if image_min_tokens > 0:
                kwargs["image_min_tokens"] = image_min_tokens

            if _MTMD:
                try:
                    import inspect

                    params = inspect.signature(handler).parameters
                    # Subclasses (Qwen3.5/3.6) may only expose **kwargs; prefer mmproj_path.
                    if "clip_model_path" in params and "mmproj_path" not in params:
                        kwargs["clip_model_path"] = mmproj_path
                    else:
                        kwargs["mmproj_path"] = mmproj_path
                except (TypeError, ValueError):
                    kwargs["mmproj_path"] = mmproj_path
            else:
                kwargs["clip_model_path"] = mmproj_path

            if handler is None:
                raise RuntimeError(
                    f'Chat handler "{chat_handler}" is not available in this llama-cpp-python build.'
                )
            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit > 0:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            # MTMD handlers (Qwen3.5/3.6 etc.) require mmproj; bind template after load.
            if handler is not None and _MTMD and issubclass(handler, MTMDChatHandler):
                cls.chat_handler = None
            elif handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None
        
        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers} (vram_limit={vram_limit}, n_ctx={n_ctx})")
        try:
            cls.llm = Llama(
                model_path,
                chat_handler=cls.chat_handler,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                use_mmap=True,
                verbose=False,
            )
        except Exception as e:
            # Re-run once with verbose so llama.cpp prints the real reason (arch/OOM/CUDA).
            print(f"[llama-cpp_vlm] Load failed ({e}); retrying with verbose=True…", flush=True)
            try:
                cls.llm = Llama(
                    model_path,
                    chat_handler=cls.chat_handler,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    use_mmap=True,
                    verbose=True,
                )
            except Exception as e2:
                raise ValueError(
                    f"Failed to load model from file: {model_path}\n"
                    f"n_gpu_layers={n_gpu_layers}, vram_limit={vram_limit}, n_ctx={n_ctx}\n"
                    f"For CPU testing set vram_limit=0 and restart ComfyUI after code changes.\n"
                    f"Original error: {e2}"
                ) from e2

        if (
            (not mmproj or mmproj == "None")
            and handler is not None
            and _MTMD
            and issubclass(handler, MTMDChatHandler)
        ):
            think_mode = "Thinking" in chat_handler
            text_handler = _text_only_chat_handler(
                handler, cls.llm, enable_thinking=think_mode
            )
            if text_handler is None:
                print(
                    f"[llama-cpp_vlm] '{chat_handler}' needs mmproj for vision; "
                    "using GGUF chat template (text-only)"
                )
            else:
                print(f"[llama-cpp_vlm] '{chat_handler}' text-only mode (no mmproj)")
                cls.chat_handler = text_handler
                cls.llm.chat_handler = text_handler

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

LLM_FOLDER_KEY = "llm"
LLM_MODEL_DIR_SETTING = "LlamaCppVlm.ModelDirectory"

def _default_llm_dir() -> str:
    return os.path.join(folder_paths.models_dir, "llm")

def _register_default_llm_folder() -> None:
    default = _default_llm_dir()
    os.makedirs(default, exist_ok=True)
    folder_paths.add_model_folder_path(LLM_FOLDER_KEY, default, is_default=True)
    paths = folder_paths.get_folder_paths(LLM_FOLDER_KEY)
    folder_paths.folder_names_and_paths[LLM_FOLDER_KEY] = (paths, {".gguf"})

def _read_model_directory_setting() -> str:
    """Read Settings override from user comfy.settings.json (if set)."""
    user_root = folder_paths.get_user_directory()
    candidates = [
        os.path.join(user_root, "default", "comfy.settings.json"),
        os.path.join(user_root, "comfy.settings.json"),
    ]
    try:
        for name in os.listdir(user_root):
            path = os.path.join(user_root, name, "comfy.settings.json")
            if os.path.isfile(path):
                candidates.append(path)
    except OSError:
        pass
    seen: set[str] = set()
    for file in candidates:
        if file in seen or not os.path.isfile(file):
            continue
        seen.add(file)
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            value = data.get(LLM_MODEL_DIR_SETTING)
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            continue
    return ""

def _resolve_configured_dir(raw: str) -> str:
    raw = os.path.expanduser(raw.strip().strip('"').strip("'"))
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    base_path = getattr(folder_paths, "base_path", None) or os.path.dirname(folder_paths.models_dir)
    candidates = [
        os.path.abspath(os.path.join(base_path, raw)),
        os.path.abspath(os.path.join(folder_paths.models_dir, raw)),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]

def get_llm_model_dirs() -> list[str]:
    """Directories to scan for .gguf (Settings override, then models/llm + extras)."""
    dirs: list[str] = []
    override = _read_model_directory_setting()
    if override:
        dirs.append(_resolve_configured_dir(override))
    try:
        for path in folder_paths.get_folder_paths(LLM_FOLDER_KEY):
            if path not in dirs:
                dirs.append(path)
    except Exception:
        pass
    if not dirs:
        dirs.append(_default_llm_dir())
    for path in dirs:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
    return dirs

def list_llm_ggufs() -> list[str]:
    found: set[str] = set()
    for folder in get_llm_model_dirs():
        files, _ = folder_paths.recursive_search(folder, excluded_dir_names=[".git"])
        found.update(folder_paths.filter_files_extensions(files, {".gguf"}))
    return sorted(found)

def resolve_llm_gguf(filename: str, kind: str = "Model") -> str:
    """Resolve a .gguf under the configured llm model dirs."""
    for folder in get_llm_model_dirs():
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate
        if os.path.islink(candidate) and not os.path.exists(candidate):
            raise FileNotFoundError(
                f'{kind} is a broken symlink under llm model dir: "{filename}" -> '
                f'"{os.path.realpath(candidate)}". Replace it with the real .gguf or fix the link target.'
            )
    searched = ", ".join(get_llm_model_dirs())
    raise FileNotFoundError(f'{kind} not found under llm model dirs [{searched}]: "{filename}"')

_register_default_llm_folder()

preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def _completion_text(output):
    msg = output["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        content = msg.get("reasoning_content") or ""
    return str(content).removeprefix(": ").lstrip()

def parse_json(json_str):
    json_output = json_str.strip().removeprefix("```json").removesuffix("```")
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    resized_frames = []
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    for item in json:
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    for item in json:
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = list_llm_ggufs()
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 512, "max": 327680, "step": 128,
                "tooltip": "Context length limit. Use 512–1024 for CPU smoke tests."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB.\n-1 = no limit (all layers on GPU)\n0 = CPU only (n_gpu_layers=0)\n>0 = approximate VRAM cap in GB"
            }),
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm-fork"
    
    '''
    @classmethod
    def IS_CHANGED(s, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        config_str = json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
        return config_str
    '''
    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Read one image at a time\nimages:  \tRead all images at once\nvideo:  \tTreat the input images as video"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video.\n(for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size of input images in "images" and "video" modes.'
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
                "queue_handler": (any_type, {"tooltip": "Used to control the execution order of instruct nodes."}),
            },
            
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def sanitize_messages(self, messages):
        clean_messages = messages.copy()
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, queue_handler=None):
        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
            #raise RuntimeError("The model has been unloaded or failed to load!")
        
        if parameters is None:
            parameters = {
                "max_tokens": 1024,
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0.05,
                "typical_p": 1.0,
                "temperature": 0.8,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 1.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0
            }
        
        _parameters = parameters.copy()
        _uid = _parameters.pop("state_uid", None)
        # Mutating the shared parameters dict breaks later runs; strip only on our copy.
        _h = LLAMA_CPP_STORAGE.chat_handler
        if _MTMD and MTMDChatHandler is not None and isinstance(_h, MTMDChatHandler):
            _parameters.pop("presence_penalty", None)
        uid = str(unique_id).rpartition('.')[-1] if _uid in (None, -1) else _uid
        # llama.cpp seeds are typically 32-bit; ComfyUI seeds can be larger.
        seed = int(seed) & 0xFFFFFFFF
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            LLAMA_CPP_STORAGE.clean_state(uid)
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts

        if save_states and last_sys_prompt == system_prompts:
            try:
                print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                messages = list(LLAMA_CPP_STORAGE.messages.get(f"{uid}", []))
            except Exception:
                messages = []
        else:
            messages = []

        # Always restore system prompt when starting a fresh turn (incl. save_states=False
        # after a prior save_states=True run with the same system text).
        if not messages and system_prompts.strip():
            messages.append({"role": "system", "content": system_prompts})
        out1 = ""
        out2 = []
        user_content = []
        if custom_prompt.strip() and "*" not in preset_prompt:
            user_content.append({"type": "text", "text": custom_prompt})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            user_content.append({"type": "text", "text": p})
            
        if images is not None:
            _mmproj = (
                (getattr(_h, "mmproj_path", None) or getattr(_h, "clip_model_path", None))
                if _h else None
            )
            if not _mmproj:
                raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")
                
            frames = images
            if video_input:
                n = len(images)
                if n <= 0:
                    raise ValueError("Video mode requires at least one image frame.")
                indices = np.linspace(0, n - 1, min(max_frames, n), dtype=int)
                frames = [images[i] for i in indices]
                
            if inference_mode == "one by one":
                tmp_list = []
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": ""}
                }
                user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
                print(f"[llama-cpp_vlm] Start processing {len(frames)} images")
                
                for i, image in enumerate(cqdm(frames)):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    for item in user_content:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = f"data:image/jpeg;base64,{data}"
                            break
                    output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                    text = _completion_text(output)
                    out2.append(text)
                    if len(frames) > 1:
                        tmp_list.append(f"====== Image {i+1} ======")
                    tmp_list.append(text)
                    
                out1 = "\n\n".join(tmp_list)
            else:
                for image in frames:
                    if len(frames) > 1:
                        data = image2base64(scale_image(image, max_size))
                    else:
                        data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                    }
                    user_content.append(image_content)
                    
                messages.append({"role": "user", "content": user_content})
                output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                out1 = _completion_text(output)
                out2 = [out1]
        else:
            messages.append({"role": "user", "content": user_content})
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
            out1 = _completion_text(output)
            out2 = [out1]
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            #LLAMA_CPP_STORAGE.states[f"{uid}"] = LLAMA_CPP_STORAGE.llm.save_state()
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)
                
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            # MTP / hybrid Qwen3.5–3.6 need an explicit KV/hybrid reset between turns.
            _handler_name = (LLAMA_CPP_STORAGE.current_config or {}).get("chat_handler", "")
            if any(h in _handler_name for h in ("Qwen3.5", "Qwen3.6")):
                try:
                    LLAMA_CPP_STORAGE.llm.n_tokens = 0
                    LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                    if getattr(LLAMA_CPP_STORAGE.llm, "is_hybrid", False) and getattr(LLAMA_CPP_STORAGE.llm, "_hybrid_cache_mgr", None) is not None:
                        LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                except Exception as e:
                    print(f"[llama-cpp_vlm] Warning: hybrid/KV reset failed: {e}")
            
        del messages
        gc.collect()
        return (out1, out2, uid)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                #"tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "stop_words": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated stop words. Generation stops when any of these are encountered."
                }),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    def process(self, **kwargs):
        stop_words = kwargs.pop("stop_words", "")
        if stop_words.strip():
            kwargs["stop"] = [w.strip() for w in stop_words.split(",") if w.strip()]
        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)"
                }),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, json, mode, label, image=None):
        mode = mode[0]
        label = label[0]

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            
            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")
    
class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        for i, bbox in enumerate(bboxes):
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)

            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)

            if x2_c > x1_c and y2_c > y1_c:
                lx0 = x1_c - x1_exp
                ly0 = y1_c - y1_exp
                lx1 = lx0 + (x2_c - x1_c)
                ly1 = ly0 + (y2_c - y1_c)
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = local_mask_np[ly0:ly1, lx0:lx1]

            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, bboxes, image_index, bbox_index):
        if not bboxes:
            raise ValueError("bboxes is empty.")
        if image_index < 0 or image_index >= len(bboxes):
            raise IndexError(f"image_index {image_index} out of range (0..{len(bboxes)-1}).")
        frame = bboxes[image_index]
        if bbox_index == 999:
            return (frame,)
        if bbox_index < 0 or bbox_index >= len(frame):
            raise IndexError(f"bbox_index {bbox_index} out of range (0..{len(frame)-1}, or 999=all).")
        return ([frame[bbox_index]],)

# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING",),
                "default": ("STRING",),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, input, key=None, default=None):
        if isinstance(input, str):
            input = [input]

        result = {
            "any": {},
            "string": {},
            "int": {},
            "float": {},
            "boolean": {},
        }
        for i, json in enumerate(input):
            val = ""
            if key is not None and key != "":
                val = get_nested_value(json.strip().removeprefix("```json").removesuffix("```"), key, default)
            else:
                raise ValueError("Key cannot be empty!")

            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception:
                result["string"][i] = val

            try:
                result["int"][i] = int(val)
            except Exception:
                result["int"][i] = val

            try:
                result["float"][i] = float(val)
            except Exception:
                result["float"][i] = val

            try:
                result["boolean"][i] = val.lower() == "true"
            except Exception:
                result["boolean"][i] = val

        if len(result["any"]) == 1:
            result["any"] = result["any"][0]
            result["string"] = result["string"][0]
            result["int"] = result["int"][0]
            result["float"] = result["float"][0]
            result["boolean"] = result["boolean"][0]

        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def process(self, input, label=""):
        if isinstance(input, str):
            input = [input]
        label = label or ""
        
        output = []
        for value in input:
            output.append(value.strip().removeprefix(f"```{label}").removesuffix("```"))
        if len(output) == 1:
            return (output[0],)
        return (output,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": ([
                    "Qwen-Image [EN]", "Qwen-Image [JA]",
                    "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [JA]",
                    "Qwen-Image-Edit [EN]", "Qwen-Image-Edit [JA]",
                    "Qwen-Image-Edit 2509 [EN]", "Qwen-Image-Edit 2509 [JA]",
                    "Qwen-Image-Edit 2511 [EN]", "Qwen-Image-Edit 2511 [JA]",
                    "Z-Image Turbo [EN]", "Z-Image Turbo [JA]",
                    "Flux.2 T2I [EN]", "Flux.2 T2I [JA]",
                    "Flux.2 I2I [EN]", "Flux.2 I2I [JA]",
                    "Wan T2V [EN]", "Wan T2V [JA]",
                    "Wan I2V [EN]", "Wan I2V [JA]",
                    "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [JA]",
                    "Wan FLF2V [EN]", "Wan FLF2V [JA]",
                    "Anima [EN]", "Anima [JA]",
                    "Krea 2 [EN]", "Krea 2 [JA]",
                    "Krea 2 Edit [EN]", "Krea 2 Edit [JA]",
                    "MiniMax H3 [EN]", "MiniMax H3 [JA]",
                    "MiniMax H3 Ref [EN]", "MiniMax H3 Ref [JA]",
                ],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm-fork"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [JA]" | "Qwen-Image [ZH]":
                return (QWEN_IMAGE_JA if preset.endswith("[JA]") else QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [JA]" | "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_JA if preset.endswith("[JA]") else QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit [EN]" | "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit [JA]":
                return (QWEN_IMAGE_EDIT_JA,)
            case "Qwen-Image-Edit 2509 [EN]" | "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2509 [JA]":
                return (QWEN_IMAGE_EDIT_2509_JA,)
            case "Qwen-Image-Edit 2511 [EN]" | "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Qwen-Image-Edit 2511 [JA]":
                return (QWEN_IMAGE_EDIT_2511_JA,)
            case "Z-Image Turbo [EN]":
                return (ZIMAGE_TURBO_EN,)
            case "Z-Image Turbo [JA]":
                return (ZIMAGE_TURBO_JA,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I [EN]" | "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 T2I [JA]":
                return (FLUX2_T2I_JA,)
            case "Flux.2 I2I [EN]" | "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Flux.2 I2I [JA]":
                return (FLUX2_I2I_JA,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [JA]":
                return (WAN_T2V_JA,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [JA]":
                return (WAN_I2V_JA,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [JA]":
                return (WAN_I2V_EMPTY_JA,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [JA]":
                return (WAN_FLF2V_JA,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case "Krea 2 [EN]":
                return (KREA2_EN,)
            case "Krea 2 [JA]":
                return (KREA2_JA,)
            case "Anima [EN]":
                return (ANIMA_EN,)
            case "Anima [JA]":
                return (ANIMA_JA,)
            case "MiniMax H3 [EN]":
                return (MINIMAX_H3_EN,)
            case "MiniMax H3 [JA]":
                return (MINIMAX_H3_JA,)
            case "MiniMax H3 Ref [EN]":
                return (MINIMAX_H3_REF_EN,)
            case "MiniMax H3 Ref [JA]":
                return (MINIMAX_H3_REF_JA,)
            case "Krea 2 Edit [EN]":
                return (KREA2_EDIT_EN,)
            case "Krea 2 Edit [JA]":
                return (KREA2_EDIT_JA,)
            case _:
                raise ValueError(f'Unknown preset: "{preset}"')

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset / プロンプト強化プリセット",
}