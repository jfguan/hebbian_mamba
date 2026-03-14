# Project Rules

- Never kick off training runs without explicit permission from the user.
- Write code top-down: if function A calls function B, define A above B. Higher-level functions come first, lower-level helpers below.
- Use minimal section comments: `# -- <section> --`
- Declare variables close to where they're used, not at the top of a function.
- Follow the coding principles of Ilya Sutskever, Jeff Dean, and John Carmack: simple code, no unnecessary abstraction, optimize for readability and correctness, prefer straightforward solutions over clever ones.
- Docstring format for args/returns:
  ```
  """Short description.

  arg1: description.
  arg2: description.

  returns: description.
  """
  ```
