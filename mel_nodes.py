import random
import re
import math
import time
import os
import torch
import folder_paths 
import comfy.sd
import datetime
from PIL import Image, ImageSequence
from io import BytesIO
import numpy as np

MAX_OUTPUTS = 20  # 最大出力数の定数

class mel_TextSplitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": " "}),
                "max_outputs": ("INT", {"default": 5, "min": 1, "max": 999}),
                "random_select": ("BOOLEAN", {"default": False}),
                # selected_number を文字列として受け取る（例："2 5"）
                "selected_number": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("string", "number")
    FUNCTION = "process"
    CATEGORY = "Text"

    def process(self, text, delimiter, max_outputs, random_select, selected_number, seed):
        # --- テキスト前処理：行頭のコメント行および行中の "#" 以降を除去 ---
        cleaned_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if line:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        
        # --- パターンマッチングで番号指定部分を検出 ---
        pattern = re.compile(r"(\d+(?:\.\d+)*):")
        matches = pattern.finditer(text)

        tokens = []
        assigned_numbers = {}  # {トークン: [割り当てられた番号リスト]}
        manual_numbers = set()

        last_index = 0
        current_numbers = None

        for match in matches:
            start, end = match.span()
            numbers = list(map(int, match.group(1).split('.')))

            # 直前のテキストを delimiter で区切る
            if last_index < start:
                chunk = text[last_index:start].strip()
                if chunk:
                    sub_tokens = chunk.split(delimiter)
                    for sub_token in sub_tokens:
                        sub_token = sub_token.strip()
                        if sub_token:
                            tokens.append(sub_token)
                            assigned_numbers[sub_token] = current_numbers or []
            current_numbers = numbers
            manual_numbers.update(numbers)
            last_index = end  # 「数字:」の後ろから新しいトークンを探す

        # 最後の部分を処理
        if last_index < len(text):
            chunk = text[last_index:].strip()
            if chunk:
                sub_tokens = chunk.split(delimiter)
                for sub_token in sub_tokens:
                    sub_token = sub_token.strip()
                    if sub_token:
                        tokens.append(sub_token)
                        assigned_numbers[sub_token] = current_numbers or []

        # delimiter で分割されたものに番号が割り振られていない場合、小さい番号を割り振る
        available_numbers = set(range(1, len(tokens) + 1)) - manual_numbers
        num_iterator = iter(sorted(available_numbers))

        sorted_tokens = []
        sorted_numbers = []

        for token in tokens:
            if not assigned_numbers[token]:  # 番号がない場合、小さい番号を割り振る
                assigned_numbers[token] = [next(num_iterator)]
            sorted_tokens.append(token)
            sorted_numbers.append(".".join(map(str, assigned_numbers[token])))

        # --- 選択処理 ---
        # selected_number は文字列入力（例："2 5"）なので、空でない場合はその数字に該当するトークンを選ぶ
        selected_tokens = []
        selected_token_numbers = []
        if selected_number.strip():
            selected_numbers_list = selected_number.strip().split()
            for t, n in zip(sorted_tokens, sorted_numbers):
                number_list = n.split('.')
                if any(num in number_list for num in selected_numbers_list):
                    selected_tokens.append(t)
                    selected_token_numbers.append(n)
        
        # random_select が True の場合、不足分をランダムに補完する
        rng = random.Random(seed)
        if random_select:
            if len(selected_tokens) < max_outputs:
                needed = max_outputs - len(selected_tokens)
                candidates = [
                    (t, n) for t, n in zip(sorted_tokens, sorted_numbers)
                    if t not in selected_tokens
                ]
                if candidates:
                    additional = rng.sample(candidates, k=min(needed, len(candidates)))
                    for t, n in additional:
                        selected_tokens.append(t)
                        selected_token_numbers.append(n)
            if not selected_tokens:
                selected_indices = rng.sample(range(len(sorted_tokens)), min(max_outputs, len(sorted_tokens)))
                selected_tokens = [sorted_tokens[i] for i in selected_indices]
                selected_token_numbers = [sorted_numbers[i] for i in selected_indices]
        # selected_number が空かつ random_select が False の場合は、シーケンシャルモード（seed を利用して開始位置を決定）
        elif not selected_tokens:
            start_index = seed % len(sorted_tokens)
            selected_indices = [(start_index + i) % len(sorted_tokens) for i in range(max_outputs)]
            selected_tokens = [sorted_tokens[i] for i in selected_indices]
            selected_token_numbers = [sorted_numbers[i] for i in selected_indices]

        text_output = ",".join(selected_tokens)
        number_output = ",".join(selected_token_numbers)
        return (text_output, number_output)


class mel_TextSplitNode2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"multiline": True}),
                "text2": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": " "}),
                "max_outputs": ("INT", {"default": 5, "min": 1, "max": 999}),
                "random_select": ("BOOLEAN", {"default": False}),
                "selected_number1": ("STRING", {"default": ""}),
                "selected_number2": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("string", "number")
    FUNCTION = "process"
    CATEGORY = "Text"

    def process(self, text1, text2, delimiter, max_outputs, random_select, selected_number1, selected_number2, seed):
        def process_text(text):
            # 各行について、先頭が "#" の行は無視し、
            # 行中にある "#" 以降の文字列も削除してから、空行も除去する
            cleaned_lines = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if line:
                    cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)
            
            pattern = re.compile(r"(\d+(?:\.\d+)*):")
            matches = pattern.finditer(text)

            tokens = []
            assigned_numbers = {}
            manual_numbers = set()

            last_index = 0
            current_numbers = None

            for match in matches:
                start, end = match.span()
                numbers = list(map(int, match.group(1).split('.')))

                if last_index < start:
                    chunk = text[last_index:start].strip()
                    if chunk:
                        sub_tokens = chunk.split(delimiter)
                        for sub_token in sub_tokens:
                            sub_token = sub_token.strip()
                            if sub_token:
                                tokens.append(sub_token)
                                assigned_numbers[sub_token] = current_numbers or []
                current_numbers = numbers
                manual_numbers.update(numbers)
                last_index = end

            if last_index < len(text):
                chunk = text[last_index:].strip()
                if chunk:
                    sub_tokens = chunk.split(delimiter)
                    for sub_token in sub_tokens:
                        sub_token = sub_token.strip()
                        if sub_token:
                            tokens.append(sub_token)
                            assigned_numbers[sub_token] = current_numbers or []

            available_numbers = set(range(1, len(tokens) + 1)) - manual_numbers
            num_iterator = iter(sorted(available_numbers))

            sorted_tokens = []
            sorted_numbers = []

            for token in tokens:
                if not assigned_numbers[token]:
                    assigned_numbers[token] = [next(num_iterator)]
                sorted_tokens.append(token)
                sorted_numbers.append(".".join(map(str, assigned_numbers[token])))

            return sorted_tokens, sorted_numbers

        tokens1, numbers1 = process_text(text1)
        tokens2, numbers2 = process_text(text2)

        if len(tokens1) == 0 or len(tokens2) == 0:
            return ("", "")

        # 乱数生成器の初期化（text1, text2 用に seed と seed+1 を使用）
        rng1 = random.Random(seed)
        rng2 = random.Random(seed + 1)

        def select_tokens(tokens, numbers, selected_number_str, max_select, rng_instance, random_select, seed):
            selected_number_list = selected_number_str.strip().split() if selected_number_str.strip() else []
            selected = []
            for i, (t, n) in enumerate(zip(tokens, numbers)):
                number_list = n.split('.')
                if any(num in number_list for num in selected_number_list):
                    selected.append((i, t, n))
            selected.sort(key=lambda x: x[0])
            selected_tokens = [x[1] for x in selected]
            selected_numbers = [x[2] for x in selected]

            if len(selected_tokens) < max_select:
                needed = max_select - len(selected_tokens)
                if random_select:
                    remaining = [(i, t, n) for i, (t, n) in enumerate(zip(tokens, numbers)) if t not in selected_tokens]
                    if remaining:
                        additional = rng_instance.sample(remaining, k=min(needed, len(remaining)))
                        selected_tokens.extend([x[1] for x in additional])
                        selected_numbers.extend([x[2] for x in additional])
                else:
                    if selected:
                        last_index = selected[-1][0]
                    else:
                        last_index = seed % len(tokens)
                    i = (last_index + 1) % len(tokens)
                    start_i = i
                    while len(selected_tokens) < max_select:
                        if tokens[i] not in selected_tokens:
                            selected_tokens.append(tokens[i])
                            selected_numbers.append(numbers[i])
                        i = (i + 1) % len(tokens)
                        if i == start_i:
                            break

            if len(selected_tokens) < max_select:
                while len(selected_tokens) < max_select:
                    selected_tokens.append(rng_instance.choice(tokens))
                    selected_numbers.append(rng_instance.choice(numbers))

            return selected_tokens, selected_numbers

        tokens1_selected, numbers1_selected = select_tokens(tokens1, numbers1, selected_number1, max_outputs, rng1, random_select, seed)
        tokens2_selected, numbers2_selected = select_tokens(tokens2, numbers2, selected_number2, max_outputs, rng2, random_select, seed)

        paired_list = list(zip(tokens1_selected, tokens2_selected, numbers1_selected, numbers2_selected))
        rng1.shuffle(paired_list)
        shuffled_tokens = [f"{t1} {t2}" if not (t1.endswith(" ") or t2.startswith(" ")) else f"{t1}{t2}" for t1, t2, _, _ in paired_list]
        shuffled_numbers = [f"{n1} {n2}" for _, _, n1, n2 in paired_list]

        text_output = ",".join(shuffled_tokens)
        number_output = ",".join(shuffled_numbers)
        return (text_output, number_output)
    
class mel_RandomIntNode:
    counter = {}  # 各シードごとのカウンター

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_value": ("INT", {"default": 100, "min": 1, "max": 100}),  # 上限値（デフォルト100）
                "random_select": ("BOOLEAN", {"default": False}),  # ランダムモード
                "increment_mode": ("BOOLEAN", {"default": False}),  # インクリメントモード
                "output_max_value": ("BOOLEAN", {"default": False}),  # max_value のみ出力するか
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1})  # シード値
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("output_value",)
    FUNCTION = "process"
    CATEGORY = "Number"

    def process(self, max_value, random_select, increment_mode, output_max_value, seed):
        if output_max_value:
            return (max_value,)  # max_value のみ出力

        if random_select:
            seed = random.randint(0, 2**32 - 1)  # ランダム選択が有効ならシードをランダムに変更
        
        rng = random.Random(seed)

        if increment_mode:
            # インクリメントモード
            if seed not in self.counter:
                self.counter[seed] = 1  # 初回は1から
            else:
                self.counter[seed] += 1  # ＋1増加
                if self.counter[seed] > max_value:
                    self.counter[seed] = 1  # 上限に達したら1に戻る
            random_int = self.counter[seed]
        else:
            # ランダムモード
            random_int = rng.randint(1, max_value)

        return (random_int,)

class mel_TextFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "filter_number": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "filter_values": ("STRING", {"default": "3 5 7"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_text",)
    FUNCTION = "process"
    CATEGORY = "Text"

    def process(self, text, filter_number, filter_values):
        try:
            exclude_numbers = set(map(int, filter_values.split()))
        except ValueError:
            exclude_numbers = set()  # 変換失敗時は空集合

        if filter_number in exclude_numbers:
            return ("" ,)
        return (text,)


    
class ResolutionSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        resolutions = [
            "208 x 368",  "272 x 368", "320 x 768", "480 x 640", "480 x 832", "480 x 1152", "512 x 512", "512 x 768", "640 x 1536", "720 x 1280", "768 x 768", "768 x 1024", "1024 x 1024", "1080 x 1920"
        ]
        return {
            "required": {
                "resolution": (resolutions, ),
                "switch": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "set_resolution"
    CATEGORY = "Custom"

    def set_resolution(self, resolution, switch):
        width, height = map(int, resolution.split(' x '))
        if switch:
            width, height = height, width
        return (width, height)

class SplitImageBatch:
    """
    Splits an input image batch into two batches: first half and second half.
    """
    CATEGORY = "image/batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Input image batch tensor [B, H, W, C]
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")  # Two output batches
    FUNCTION = "split_batch"
    OUTPUT_NODE = False
    RETURN_NAMES = ("first_batch", "second_batch")

    def split_batch(self, images):
        """
        Splits the input tensor along the batch dimension into two halves.
        If the batch size is odd, the second batch will have one extra image.
        Args:
            images (torch.Tensor): Tensor of shape [B, H, W, C]
        Returns:
            tuple: (first_half, second_half) as two tensors of shape [B1, H, W, C] and [B2, H, W, C]
        """
        batch_size = images.shape[0]
        half = batch_size // 2
        first = images[:half]
        second = images[half:]
        return (first, second)


class AddFileNameonly:
    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
                "file_name": ("STRING", {"multiline": False, "default": ""}),
                "add_date_time": (["disable", "prefix", "postfix", "posnot_"],),
                "date_time_format": ("STRING", {"multiline": False, "default": "%Y_%m_%d_%H:%M:%S"}),
            }
         }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "add_filename_prefix"
    CATEGORY = "filename"

    def add_filename_prefix(self, file_name, add_date_time, date_time_format):
        
        if add_date_time == "disable":
            new_name = file_name
        else:
            date_str = datetime.datetime.now().strftime(date_time_format)
            if add_date_time == "prefix":
                new_name = date_str + "_" + file_name
            elif add_date_time == "postfix":
                new_name = file_name + "_" + date_str
            elif add_date_time == "posnot_":
                new_name = file_name + date_str
            else:
                new_name = file_name

        new_path = os.path.join(new_name)
        return (new_path,)
    
    
#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    
class UnetSelector_gguf:
    RETURN_TYPES = (folder_paths.get_filename_list("unet_gguf"),)
    RETURN_NAMES = ("unet_name",)
    FUNCTION = "get_names" 
    CATEGORY = 'ImageSaverTools/utils'

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("unet_gguf"), ),}}

    def get_names(self, unet_name):
        return (unet_name,)     
    
    
    
    
    
    

# ノードマッピングの登録
NODE_CLASS_MAPPINGS = {
    "mel_TextSplitNode": mel_TextSplitNode,
    "mel_TextSplitNode2": mel_TextSplitNode2,
    "mel_RandomIntNode": mel_RandomIntNode,
    "ResolutionSwitcher": ResolutionSwitcher,
    "mel_TextFilterNode": mel_TextFilterNode,
    "Unet Selector_gguf": UnetSelector_gguf,
    "AddFileNameonly": AddFileNameonly,
    "Split Image Batch": SplitImageBatch,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    #"mel_TextSplitNode": "mel_Text Split with Selection & Indexing",
    #"mel_TextSplitNode2": "mel_Text Split with Dual Selection & Indexing2",
    #"mel_RandomIntNode": "mel_Random Integer with Modes",
    
}
