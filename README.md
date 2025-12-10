# ComfyUI_Custom_Nodes_Agent

Agent files for working with custom nodes inside ComfyUI workspace in VS Code primarily.

## Instructions

### Make a clone of ComfyUI on your drive.

Optional: Set up the venv as usual.

Drop the three files in the top level directory of that ComfyUI clone.

Create/Clone your custom node folder/repository inside the custom_node directory.

### File Structure

```plaintext
├─┐ ComfyUI/
│ └─┐ custom_nodes/
│   └── your_custom_node/
│── AGENTS.md
├── python.instructions.md
└── custom_nodes.examples.md
```

### Open VS Code and select Open Folder.

Select the ComfyUI folder to open.

Open the file tree sidebar
Right click your custom nodes folder and select "Add Folder to Chat" if using Copilot Chat
or "Add Files as Context" if using Continue or any equivalent of any other agentic chat.

### Considerations

Not been tested using any of the cli tools but I feel that it should work if you just specify the directory
of your custom node as the one to edit or if it is the only one there.

### Purpose

The instructions should prevent it from trying to edit ComfyUI core files and only edit your custom node.