;;; magic-todo-org.el --- Magic ToDo (goblin.tools-style) in Org -*- lexical-binding: t; -*-

;; Local-first task breakdown UI for Org buffers.
;; Shells out to scripts/magic_todo_mlx.py (MLX-LM) and inserts an Org checklist.

(require 'json)
(require 'subr-x)
(require 'cl-lib)

(defgroup magic-todo-org nil
  "Magic ToDo task breakdown in Org."
  :group 'org)

(defun magic-todo-org--clean-python-env ()
  "Return `process-environment' with PYTHONPATH and PYTHONHOME removed.
Prevents flox/nix system packages from shadowing the venv."
  (cl-remove-if (lambda (e)
                  (or (string-prefix-p "PYTHONPATH=" e)
                      (string-prefix-p "PYTHONHOME=" e)))
                process-environment))

(defun magic-todo-org--walk-up-for (filename &optional start)
  "Walk up from START looking for FILENAME, return full path or nil."
  (let ((dir (or start default-directory)))
    (cl-block nil
      (while dir
        (let ((candidate (expand-file-name filename dir)))
          (when (file-exists-p candidate)
            (cl-return candidate)))
        (let ((parent (file-name-directory (directory-file-name dir))))
          (if (equal parent dir)
              (cl-return nil)
            (setq dir parent)))))))

(defun magic-todo-org--resolve-python ()
  "Find .venv-mlx-lm/bin/python, checking `default-directory' first, then load path."
  (or (magic-todo-org--walk-up-for ".venv-mlx-lm/bin/python" default-directory)
      (magic-todo-org--walk-up-for
       ".venv-mlx-lm/bin/python"
       (file-name-directory (or load-file-name buffer-file-name "")))
      magic-todo-org-python))

(defun magic-todo-org--resolve-script ()
  "Find scripts/magic_todo_mlx.py, checking `default-directory' first, then load path."
  (or (magic-todo-org--walk-up-for "scripts/magic_todo_mlx.py" default-directory)
      (magic-todo-org--walk-up-for
       "scripts/magic_todo_mlx.py"
       (file-name-directory (or load-file-name buffer-file-name "")))
      magic-todo-org-script))

(defcustom magic-todo-org-python "python3"
  "Fallback Python for Magic ToDo. Normally auto-detected from .venv-mlx-lm."
  :type 'string)

(defcustom magic-todo-org-script "magic_todo_mlx.py"
  "Fallback path to magic_todo_mlx.py. Normally auto-detected."
  :type 'string)

(defcustom magic-todo-org-default-model "mlx-community/Qwen3-8B-4bit"
  "Default MLX model repo id."
  :type 'string)

(defcustom magic-todo-org-default-spice 3
  "Default spiciness (1-5)."
  :type 'integer)

(defcustom magic-todo-org-default-max-tokens 900
  "Default max tokens for generation."
  :type 'integer)

(defcustom magic-todo-org-roam-directory nil
  "Directory to create org-roam notes in (nil means: use `org-roam-directory` if available, else ~/org-roam/)."
  :type '(choice (const :tag "Auto" nil) directory))

(defcustom magic-todo-org-temp 0.2
  "Sampling temperature."
  :type 'number)

(defcustom magic-todo-org-top-p 0.9
  "Nucleus sampling top_p."
  :type 'number)

(defcustom magic-todo-org-store-settings-in-properties t
  "If non-nil, store model/spice/task in Org properties for refresh."
  :type 'boolean)

(defun magic-todo-org--assert-exec ()
  (let ((py (magic-todo-org--resolve-python))
        (sc (magic-todo-org--resolve-script)))
    (unless (file-exists-p py)
      (user-error "Missing python: %s (create .venv-mlx-lm with: python3 -m venv .venv-mlx-lm && .venv-mlx-lm/bin/pip install -r requirements.txt)" py))
    (unless (file-exists-p sc)
      (user-error "Missing script: %s" sc))))

(defun magic-todo-org--gather-context ()
  "Collect existing Magic ToDo breakdowns from the current buffer as context.
Returns a string of sibling headings with their checklists, or nil."
  (when (derived-mode-p 'org-mode)
    (save-excursion
      (let ((parts nil)
            (limit 2000))
        (goto-char (point-min))
        (while (re-search-forward "^\\*+ \\(TODO \\|DONE \\)?" nil t)
          (let* ((beg (line-beginning-position))
                 (end (save-excursion (org-end-of-subtree t t) (point)))
                 (text (buffer-substring-no-properties beg (min end (+ beg 500)))))
            (push text parts)))
        (when parts
          (let ((ctx (string-join (nreverse parts) "\n")))
            (if (> (length ctx) limit)
                (substring ctx 0 limit)
              ctx)))))))

(defun magic-todo-org--call-json (task spice model)
  "Return parsed JSON object from generator for TASK.
Runs asynchronously with a spinner so Emacs stays responsive."
  (magic-todo-org--assert-exec)
  (let* ((python (magic-todo-org--resolve-python))
         (script (magic-todo-org--resolve-script))
         (context (magic-todo-org--gather-context))
         (process-environment (magic-todo-org--clean-python-env))
         (args (append (list script
                             "--format" "json"
                             "--spice" (number-to-string spice)
                             "--model" model
                             "--max-tokens" (number-to-string magic-todo-org-default-max-tokens)
                             "--temp" (number-to-string magic-todo-org-temp)
                             "--top-p" (number-to-string magic-todo-org-top-p))
                       (when context (list "--context" context))))
         (outbuf (generate-new-buffer " *magic-todo-out*"))
         (errbuf (generate-new-buffer " *magic-todo-err*"))
         (done nil)
         (exit-code nil)
         (reporter (make-progress-reporter "Magic ToDo: generating...")))
    (unwind-protect
        (progn
          (let ((proc (make-process
                       :name "magic-todo"
                       :buffer outbuf
                       :stderr errbuf
                       :command (cons python args)
                       :sentinel (lambda (_p _e) (setq done t)))))
            (process-send-string proc task)
            (process-send-eof proc)
            (while (not done)
              (progress-reporter-update reporter)
              (accept-process-output proc 0.2))
            (setq exit-code (process-exit-status proc)))
          (progress-reporter-done reporter)
          (unless (equal exit-code 0)
            (let ((stderr (with-current-buffer errbuf (buffer-string)))
                  (stdout (with-current-buffer outbuf (buffer-string))))
              (user-error "Magic ToDo failed (exit %s)\n\nstdout:\n%s\n\nstderr:\n%s"
                          exit-code stdout (or stderr ""))))
          (with-current-buffer outbuf
            (let* ((raw (buffer-string))
                   (start (string-match "{" raw)))
              (unless start
                (user-error "Magic ToDo produced no JSON.\n\nOutput:\n%s" raw))
              (let ((depth 0)
                    (end nil)
                    (i start))
                (while (and (< i (length raw)) (not end))
                  (let ((ch (aref raw i)))
                    (cond
                     ((= ch 123) (setq depth (1+ depth)))  ; {
                     ((= ch 125) (setq depth (1- depth))    ; }
                      (when (= depth 0) (setq end (1+ i))))))
                  (setq i (1+ i)))
                (unless end
                  (user-error "Magic ToDo produced incomplete JSON.\n\nOutput:\n%s" raw))
                (let ((json-object-type 'alist)
                      (json-array-type 'list)
                      (json-false nil)
                      (json-null nil))
                  (json-read-from-string (substring raw start end)))))))
      (kill-buffer errbuf)
      (kill-buffer outbuf))))

(defun magic-todo-org--cached-models ()
  (magic-todo-org--assert-exec)
  (let ((process-environment (magic-todo-org--clean-python-env)))
    (with-temp-buffer
      (let ((exit (process-file (magic-todo-org--resolve-python) nil t nil
                                (magic-todo-org--resolve-script) "--list-models")))
        (if (equal exit 0)
            (split-string (string-trim (buffer-string)) "\n" t)
          nil)))))

(defun magic-todo-org--read-spice ()
  (let* ((choices '("1" "2" "3" "4" "5"))
         (s (completing-read
             (format "Spice (1-5, default %d): " magic-todo-org-default-spice)
             choices nil t nil nil (number-to-string magic-todo-org-default-spice))))
    (string-to-number s)))

(defun magic-todo-org--read-model ()
  (let* ((models (magic-todo-org--cached-models))
         (default magic-todo-org-default-model)
         (prompt (format "Model (default %s): " default)))
    (if (and models (member default models))
        (completing-read prompt models nil t nil nil default)
      (read-string prompt nil nil default))))

(defun magic-todo-org--insert-checklist (steps)
  (dolist (step steps)
    (let* ((text (string-trim (or (alist-get 'text step) "")))
           (substeps (alist-get 'substeps step)))
      (when (string-empty-p text)
        (setq text "Step"))
      (insert (format "- [ ] %s\n" text))
      (when (listp substeps)
        (dolist (ss substeps)
          (let ((sst (string-trim (or (alist-get 'text ss) ""))))
            (unless (string-empty-p sst)
              (insert (format "  - [ ] %s\n" sst)))))))))

(defun magic-todo-org--props-get (key)
  (when (derived-mode-p 'org-mode)
    (org-entry-get (point) key t)))

(defun magic-todo-org--props-put (key val)
  (when (and magic-todo-org-store-settings-in-properties
             (derived-mode-p 'org-mode))
    (org-entry-put (point) key val)))

(defun magic-todo-org--heading-text ()
  (string-trim
   (or (magic-todo-org--props-get "MAGIC_TODO_TASK")
       (nth 4 (org-heading-components))
       "")))

(defun magic-todo-org--read-spice-maybe (prompt default)
  (let* ((choices '("1" "2" "3" "4" "5"))
         (s (completing-read prompt choices nil t nil nil (number-to-string default))))
    (string-to-number s)))

(defun magic-todo-org--read-model-maybe (prompt default)
  (let ((models (magic-todo-org--cached-models)))
    (if (and models (member default models))
        (completing-read prompt models nil t nil nil default)
      (read-string prompt nil nil default))))

(defun magic-todo-org--delete-subtree-body ()
  "Delete everything inside the current heading, leaving the heading and properties."
  (save-excursion
    (org-back-to-heading t)
    (let ((start (progn
                   (forward-line 1)
                   (when (looking-at-p "^[ \t]*:PROPERTIES:")
                     (re-search-forward "^[ \t]*:END:[ \t]*$" nil t)
                     (forward-line 1))
                   (point)))
          (end (progn (org-end-of-subtree t t) (point))))
      (when (< start end)
        (delete-region start end)
        (goto-char start)))))

(defun magic-todo-org--goto-subtree-body-start ()
  "Move point to where checklist content should be inserted for this heading."
  (org-back-to-heading t)
  (forward-line 1)
  (when (looking-at-p "^[ \t]*:PROPERTIES:")
    (re-search-forward "^[ \t]*:END:[ \t]*$" nil t)
    (forward-line 1)))

(defun magic-todo-org--slugify (s)
  (let* ((s (downcase (string-trim s)))
         (s (replace-regexp-in-string "[^a-z0-9]+" "-" s))
         (s (replace-regexp-in-string "^-+" "" s))
         (s (replace-regexp-in-string "-+$" "" s)))
    (if (string-empty-p s) "magic-todo" s)))

;;;###autoload
(defun magic-todo-org-insert (task spice model)
  "Insert a Magic ToDo breakdown as an Org subtree at point.

TASK is the natural language task.
SPICE is 1-5.
MODEL is an MLX model repo id (e.g. mlx-community/Qwen3-4B-4bit)."
  (interactive
   (list (read-string "Task: " (when (use-region-p)
                                (buffer-substring-no-properties (region-beginning) (region-end))))
         (magic-todo-org--read-spice)
         (magic-todo-org--read-model)))
  (unless (derived-mode-p 'org-mode)
    (user-error "Not an Org buffer"))
  (let* ((plan (magic-todo-org--call-json task spice model))
         (title (string-trim (or (alist-get 'title plan) task)))
         (steps (alist-get 'steps plan)))
    (unless (listp steps)
      (user-error "Bad plan: missing steps"))
    (org-insert-heading-respect-content)
    (insert (format "TODO %s" title))
    (org-schedule nil (format-time-string "%Y-%m-%d"))
    (org-end-of-meta-data t)
    (insert "\n")
    (magic-todo-org--insert-checklist steps)
    (when (use-region-p)
      (delete-region (region-beginning) (region-end)))))

;;;###autoload
(defun magic-todo-org-refresh-at-point (&optional force-prompt)
  "Regenerate the checklist under the current Org heading.

Uses heading text or MAGIC_TODO_TASK property as the task.
If MAGIC_TODO_MODEL / MAGIC_TODO_SPICE properties are set, reuse them.
With prefix argument FORCE-PROMPT, always prompt for model/spice/task."
  (interactive "P")
  (unless (derived-mode-p 'org-mode)
    (user-error "Not an Org buffer"))
  (save-excursion
    (org-back-to-heading t)
    (let* ((task0 (magic-todo-org--heading-text))
           (task (if force-prompt
                     (read-string "Task: " task0)
                   task0))
           (spice0 (or (when-let ((s (magic-todo-org--props-get "MAGIC_TODO_SPICE")))
                         (ignore-errors (string-to-number s)))
                       magic-todo-org-default-spice))
           (spice (if force-prompt
                      (magic-todo-org--read-spice-maybe
                       (format "Spice (1-5, default %d): " spice0) spice0)
                    spice0))
           (model0 (or (magic-todo-org--props-get "MAGIC_TODO_MODEL")
                       magic-todo-org-default-model))
           (model (if force-prompt
                      (magic-todo-org--read-model-maybe
                       (format "Model (default %s): " model0) model0)
                    model0)))
      (when (string-empty-p (string-trim task))
        (user-error "No task found (heading is empty)."))
      (magic-todo-org--props-put "MAGIC_TODO_TASK" task)
      (magic-todo-org--props-put "MAGIC_TODO_SPICE" (number-to-string spice))
      (magic-todo-org--props-put "MAGIC_TODO_MODEL" model)
      (let* ((plan (magic-todo-org--call-json task spice model))
             (steps (alist-get 'steps plan)))
        (unless (listp steps)
          (user-error "Bad plan: missing steps"))
        (magic-todo-org--delete-subtree-body)
        (magic-todo-org--goto-subtree-body-start)
        (magic-todo-org--insert-checklist steps)
        (save-buffer)))))

;;;###autoload
(defun magic-todo-org-roam-new (task spice model)
  "Create a new org-roam note containing a Magic ToDo breakdown."
  (interactive
   (list (read-string "Task: " (when (use-region-p)
                                (buffer-substring-no-properties (region-beginning) (region-end))))
         (magic-todo-org--read-spice)
         (magic-todo-org--read-model)))
  (let* ((plan (magic-todo-org--call-json task spice model))
         (title (string-trim (or (alist-get 'title plan) task)))
         (steps (alist-get 'steps plan))
         (dir (cond
               ((and magic-todo-org-roam-directory (file-directory-p magic-todo-org-roam-directory))
                (expand-file-name magic-todo-org-roam-directory))
               ((and (boundp 'org-roam-directory) (stringp org-roam-directory))
                (expand-file-name org-roam-directory))
               (t (expand-file-name "~/org-roam/"))))
         (fname (format "%s-%s.org"
                        (format-time-string "%Y%m%d%H%M%S")
                        (magic-todo-org--slugify title)))
         (path (expand-file-name fname dir)))
    (unless (file-directory-p dir)
      (make-directory dir t))
    (find-file path)
    (when (= (buffer-size) 0)
      (insert "#+title: " title "\n")
      (insert "#+created: " (format-time-string "[%Y-%m-%d %a %H:%M]") "\n\n")
      (insert (format "* TODO %s\n" title))
      (insert (format "SCHEDULED: %s\n" (format-time-string "<%Y-%m-%d %a>")))
      (magic-todo-org--insert-checklist steps))
    (save-buffer)))

(defface magic-todo-org-done-face
  '((t :strike-through t :foreground "gray50"))
  "Face for checked-off checkbox items."
  :group 'magic-todo-org)

(defface magic-todo-org-divider-face
  '((t :foreground "gold" :weight bold :height 1.3 :underline t))
  "Face for group divider headings (lines starting with * ───)."
  :group 'magic-todo-org)

(defun magic-todo-org--fontify-checkboxes ()
  "Add font-lock rules to strike through checked checkbox items."
  (font-lock-add-keywords nil
    '(("^[ \t]*- \\[X\\] \\(.*\\)$" 1 'magic-todo-org-done-face t))
    'append))

(defun magic-todo-org--fontify-dividers ()
  "Add divider heading rule via org's own keyword mechanism."
  (push '("^\\*+ ───.*$" 0 'magic-todo-org-divider-face t)
        org-font-lock-extra-keywords))

(add-hook 'org-font-lock-set-keywords-hook #'magic-todo-org--fontify-dividers)

(defun magic-todo-org--setup ()
  "Set up magic-todo features in Org buffers."
  (magic-todo-org--fontify-checkboxes))

(add-hook 'org-mode-hook #'magic-todo-org--setup)

(provide 'magic-todo-org)
;;; magic-todo-org.el ends here
