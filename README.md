# ComfyUI-llama-cpp_vlm_fork

llama.cpp ベースで ComfyUI 上の LLM / VLM をネイティブ実行するカスタムノードです。

本リポジトリは [**lihaoyun6/ComfyUI-llama-cpp_vlm**](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) のフォークです。素晴らしい実装と継続的なメンテナンスに、心より感謝します。

バージョン **2.0.0** 以降は上流と**非互換**です（モデル配置先・ノードカテゴリ・依存解決が異なります）。上流ワークフローをそのまま使う場合は [上流リポジトリ](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) を利用してください。

ComfyUI Manager 上の表示名: **`llama-cpp_vlm-fork`**  
ノードカテゴリ: **`llama-cpp-vlm-fork`**

## プレビュー

![](./img/preview.jpg)

## このフォークについて

上流プロジェクトの機能を引き継ぎつつ、次のような変更・拡張を入れています。

- モデル配置先のデフォルトを `ComfyUI/models/llm` に変更（Settings で変更可。上流の `models/LLM` とは非互換）
- 起動時に `prestartup_script.py` が OS / Python 向けの [JamePeng llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) wheel を自動導入（未インストール時）
- Linux / Windows では同スクリプトが不足時に CUDA ランタイムを自動導入し、**他ノードが `llama_cpp` を import する前に** ライブラリ探索パスを通す（後からでは CUDA バックエンドが登録されず CPU のままになる）
- Qwen3.6 など新しい chat handler / MTP 系モデル向けの調整
- llama-cpp-python 0.3.46 の `mmproj_path` 対応、CPU（`vram_limit=0` + mmap）動作など
- Prompt Enhancer に Krea 2 / Krea 2 Edit / Anima / MiniMax H3 など向けプリセット（英語・日本語）を追加

## インストール

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mckey-dev/ComfyUI-llama-cpp_vlm_fork.git
```

ComfyUI 起動時、`prestartup_script.py` が不足分を自動インストールします。

- 全 OS: `llama-cpp-python`（`requirements.txt` の platform wheel）
- Linux: `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` / `nvidia-nccl-cu12`（`.so` が無いとき）
- Windows: `nvidia-cuda-runtime-cu13` / `nvidia-cublas-cu13`（`+cu130` wheel 用 DLL が無いとき）

**補足:** 新UI（Registry）は `pyproject.toml` の依存だけを入れます。`llama-cpp-python` と上記 CUDA ランタイムは入りません（旧 Manager でも URL 直指定 wheel は失敗しやすいです）。事前に入れたい場合:

```bash
python -m pip install -r ComfyUI-llama-cpp_vlm_fork/requirements.txt
# Linux GPU:
python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nccl-cu12
# Windows GPU (+cu130):
python -m pip install nvidia-cuda-runtime-cu13 nvidia-cublas-cu13
```

自動／手動インストールに失敗した場合はソースビルドを試してください。

```bash
pip uninstall llama-cpp-python -y
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --no-cache-dir
```

### モデル

`.gguf` を `ComfyUI/models/llm` に配置してください（初回起動時にフォルダを作成します）。

ComfyUI の **Settings → llama-cpp-vlm-fork → GGUF model directory** でパスを変更できます（空欄 = デフォルト）。絶対パス、または `models` フォルダからの相対パス（例: `llm`）。Settings にフォルダ選択 UI は無いのでパスを手入力してください。変更後は Loader を開き直すか画面をリロードしてください。

追加ディレクトリは `extra_model_paths.yaml` の `llm:` キーでも指定できます。

> VLM で画像入力する場合は、対応する `mmproj` も同じフォルダへ置き、Loader で選択してください。

## 謝辞

- [lihaoyun6/ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) @lihaoyun6 — 本フォークの上流。開発と公開に深く感謝します
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
