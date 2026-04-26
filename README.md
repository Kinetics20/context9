## Pre-commit

Install and run pre-commit from the project's dev environment:

```sh
uv sync --dev
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
uv run pre-commit run --all-files
```

The hooks themselves use `uv run --frozen`, so Python commands run with the interpreter selected for
this project, for example the version pinned in `.python-version`.

```mermaid
flowchart TD
    A[Start chunk_text text, max_chars, overlap] --> B[Validate options]
    B --> C{Invalid options?}
    C -- yes --> X[Raise ValueError]
    C -- no --> D[Normalize text]
    D --> E[Split by blank lines into paragraphs]
    E --> F[Initialize chunks = empty list]
    F --> G[Initialize current = empty string]
    G --> H{More paragraphs?}

    H -- no --> I{current not empty?}
    I -- yes --> J[Append current to chunks]
    I -- no --> K[Return chunks]
    J --> K

    H -- yes --> L[Take next paragraph]
    L --> M{paragraph empty?}
    M -- yes --> H
    M -- no --> N{paragraph length > max_chars?}

    N -- yes --> O{current not empty?}
    O -- yes --> P[Append current to chunks]
    P --> Q[Set current = empty]
    O -- no --> R[Split large paragraph into max_chars blocks with overlap]
    Q --> R
    R --> S[Extend chunks with split blocks]
    S --> H

    N -- no --> T[Candidate = current + blank line + paragraph]
    T --> U{candidate length <= max_chars?}

    U -- yes --> V[Set current = candidate]
    V --> H

    U -- no --> W[Append current to chunks]
    W --> Y[Take overlap tail from current]
    Y --> Z[Set current = overlap tail + blank line + paragraph]
    Z --> AA{current length <= max_chars?}

    AA -- yes --> H
    AA -- no --> AB[Set current = paragraph]
    AB --> H

```