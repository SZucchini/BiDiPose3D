# commit rules
- One commit should only contain one logical change.
- If you need to make multiple changes, please create separate commits for each change.
- If current branch is the main branch, please checkout a new branch from the main branch and make changes in that branch.
- Please push the changes to the remote after commit.

# message format
- The commit message should be in English.
- The commit message should be in the following format:
  ```
  <type>: <subject>

  <body>
  ```

## type
- The type should be one of the following:
  - feat: A new feature
  - fix: A bug fix
  - docs: Documentation only changes
  - style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  - refactor: A code change that neither fixes a bug nor adds a feature
  - perf: A code change that improves performance
  - test: Adding missing or correcting existing tests
  - chore: Changes to the build process or auxiliary tools and libraries such as documentation generation

## subject
- The subject should be a short summary of the change.

## body
- The body should be a more detailed explanation of the change.
- The body should include the motivation for the change and contrast this with previous behavior.
