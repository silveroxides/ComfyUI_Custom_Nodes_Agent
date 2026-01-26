# ComfyUI Custom Node Development Agents Instructions

This repository provides a specialized set of instructions and examples designed to guide AI coding agents (such as GitHub Copilot, Continue, or custom IDE agents) in developing, debugging, and documenting custom nodes for ComfyUI.

> [!NOTE]
> **IDE Agnostic:** While these instructions are often used within VS Code, they are not limited to any specific IDE or platform. They can be integrated into any workflow that supports providing system instructions or additional context to an AI agent (e.g., Cursor, Zed, Windsurf, or Claude Code etc.).

## Purpose

The primary goal of this repository is to establish a strict operational boundary for AI agents working within a ComfyUI workspace. These instructions:
- **Enforce Scope:** Prevent agents from modifying ComfyUI core files while allowing full access to the `custom_nodes/` directory.
- **Provide Context:** Supply agents with the necessary technical knowledge regarding ComfyUI's V1 and V3 APIs, tensor formats, and device management.
- **Standardize Development:** Ensure that generated code follows PEP 8 standards and ComfyUI-specific best practices.

## Included Instruction Files

- [`AGENTS.md`](AGENTS.md): The core rulebook for the agent, defining file system permissions, search strategies, and API requirements.
- [`python.instructions.md`](python.instructions.md): Python-specific coding standards and style guidelines.
- [`custom_node.examples.md`](custom_node.examples.md): A comprehensive library of tagged code examples for various ComfyUI operations (V3 API, latent manipulation, hooks, etc.).

## Setup Instructions

### 1. Prepare your ComfyUI Environment
Ensure you have a local clone of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

### 2. Install Instruction Files
Copy the following files from this repository into the **root directory** of your ComfyUI installation:
- `AGENTS.md`
- `python.instructions.md`
- `custom_node.examples.md`

### 3. File Structure
Your ComfyUI directory should look like this:
```plaintext
├─┐ ComfyUI/
│ ├─┐ custom_nodes/
│ │ └── your_custom_node_repo/
│ ├── AGENTS.md
│ ├── python.instructions.md
│ └── custom_node.examples.md
│ └── ... (core ComfyUI files)
```

### 4. Configure your AI Agent
1. Open the ComfyUI root folder in VS Code (or your preferred agent-enabled IDE).
2. Point your agent to these instruction files:
   - **GitHub Copilot Chat:** Right-click the `AGENTS.md`, `python.instructions.md`, and `custom_node.examples.md` files and select "Add to Chat".
   - **Continue / Custom Agents:** Add these files to your context or "rules" configuration for the project.
   - **Specific Mode Agents:** If using an agent that supports system prompts or modes (like Gemini), provide the contents of `AGENTS.md` as the system instruction.

## Considerations

- **Registration Requirement:** Custom nodes are registered at runtime. AI agents cannot verify imports via simple terminal commands; testing must be done by running ComfyUI.
- **V3 API Preference:** These instructions encourage the use of the modern V3 API (`io.ComfyNode`, `io.Schema`) for all new developments.
