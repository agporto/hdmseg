---
name: Bug report
about: Report incorrect results, a crash, or unexpected behavior
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
A minimal, self-contained example (Python or Rust) that triggers the issue.
Please include the stack shape `(N, M, D)` and any non-default parameters.

```python
import numpy as np
import hdmseg
# ...
```

**Expected behavior**
What you expected to happen instead.

**Environment**
- hdmseg version:
- Installed via: [ ] GitHub Release wheel  [ ] built from source  [ ] Rust crate
- OS and architecture:
- Python version (if applicable):

**Additional context**
Anything else that might help — number of specimens/loci, the `select`
method and whether `n_neighbors` changes the outcome.
