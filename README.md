# Magic ToDo Org (Local MLX)

Terminal + Emacs Org-mode clone of goblin.tools "Magic ToDo", running locally on Apple Silicon using MLX.

## Contents

- `scripts/magic_todo_mlx.py`: CLI that generates a task breakdown using an MLX-LM model.
- `emacs/magic-todo-org.el`: Emacs integration for inserting and refreshing checklists in Org headings.

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- `mlx` + `mlx-lm` installed in whatever Python you run the CLI with
- An MLX-LM model available (cached under HuggingFace cache, e.g. `mlx-community/Qwen3-4B-4bit`)

## CLI Usage

```bash
python scripts/magic_todo_mlx.py --list-models

python scripts/magic_todo_mlx.py --spice 3 "delete extra key in tailscale"

echo "phone calling bot to call for quote" | python scripts/magic_todo_mlx.py --format md --spice 4
```

## Emacs Setup

Add to `~/.emacs.d/init.el`:

```elisp
(require 'transient)
(add-to-list 'load-path "/ABS/PATH/TO/magic-todo-org/emacs")
(require 'magic-todo-org)
```

### Commands

- `M-x magic-todo-org-insert`
  - Prompts for task/spice/model, inserts a new heading + checklist.
- `M-x magic-todo-org-refresh-at-point`
  - Regenerates checklist under current heading.
  - Stores/uses these properties on the heading:
    - `MAGIC_TODO_TASK`
    - `MAGIC_TODO_SPICE`
    - `MAGIC_TODO_MODEL`
  - Use `C-u` prefix to force prompts.

## Notes

- Some MLX-LM models emit non-JSON progress text; the Emacs code extracts the first JSON object from output.

