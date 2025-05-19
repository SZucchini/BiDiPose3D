# pull request rules
- The base branch should be the main branch.
- Use the `gh pr` command to create a pull request.
- The pull request title should be in English.
- The pull request body should be in Japanese.

# pull request command and format
- Please use the following command and format to create a pull request:
  ```
  gh pr create \
    --base main \
    --head "$current_branch" \
    --title "<type>: <subject>" \
    --body "## 変更内容

    - <変更内容1>
    - <変更内容2>

    ## 背景・目的

    - <変更の背景・目的1>
    - <変更の背景・目的2>

    ## 参考情報（あれば）
    - <参考情報>

    "
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
