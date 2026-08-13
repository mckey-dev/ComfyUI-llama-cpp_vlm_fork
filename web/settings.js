import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ComfyUI.llama_cpp_vlm_fork.Settings",
  settings: [
    {
      id: "LlamaCppVlm.ModelDirectory",
      name: "GGUF model directory",
      type: "text",
      defaultValue: "",
      category: ["llama-cpp-vlm-fork", "Models", "GGUF model directory"],
      tooltip:
        "Directory for .gguf / mmproj. Empty = ComfyUI/models/llm. Absolute path, or relative to the models folder (e.g. llm). Settings has no folder picker — type the path. Refresh / re-open the Loader after changing.",
    },
  ],
});
