# Contributing to Cramer

Thank you for considering contributing to the Cramer project! We welcome contributions of all kinds, including bug reports, feature requests, and code contributions.

## How to Contribute

If you find any issues or have suggestions for improvements, feel free to [open an issue](https://github.com/cm-jones/cramer/issues/new) or submit a [pull request](https://github.com/cm-jones/cramer/compare).

### Steps to Contribute

1. **Fork the Repository**

   - Navigate to the [Cramer repository](https://github.com/cm-jones/cramer) on GitHub.
   - Click the "Fork" button in the top-right corner of the page.

2. **Clone the Fork**

   - Clone your forked repository to your local machine:
     ```sh
     git clone https://github.com/YOUR_USERNAME/cramer.git
     cd cramer
     ```

3. **Create a New Branch**

   - Create a new branch with a descriptive name:
     ```sh
     git checkout -b descriptive-branch-name
     ```

4. **Make Your Changes**

   - Make your changes to the codebase.
   - Ensure your changes follow the project's coding standards and conventions.

5. **Commit and Push Your Changes**

   - Commit your changes with a meaningful commit message:
     ```sh
     git add .
     git commit -m "Description of the changes made"
     ```
   - Push your changes to your fork:
     ```sh
     git push origin descriptive-branch-name
     ```

6. **Create a Pull Request**

   - Navigate to your forked repository on GitHub.
   - Click the "Compare & pull request" button next to your branch.
   - Provide a detailed description of your changes and submit the pull request.

### Coding Style

To maintain code consistency, we use clang-format with the following settings:
- BasedOnStyle: Google
- IndentWidth: 4
- ColumnLimit: 80

Please ensure that your code adheres to this style.

#### Using Clang-Format

You can format your code using clang-format before committing:

```sh
clang-format -i path/to/your/file.cpp
```

Alternatively, you can set up your editor or IDE to format the code automatically on save. Refer to your editor or IDE documentation for instructions on how to integrate clang-format.

### Pre-Commit Hooks

Before you start making changes, we recommend that you install a pre-commit hook via the `scripts/install_hooks.sh` script. This will ensure that each commit is properly built, tested, linted, and formatted.

#### Installing Pre-Commit Hook

1. Make all scripts executable by running the following command in the project's root directory:
   ```sh
   chmod +x scripts/*.sh
   ```

2. Run the install hook script:
   ```sh
   ./scripts/install_hooks.sh
   ```

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and respectful community.

## Getting Help

If you need any help, feel free to [open an issue](https://github.com/cm-jones/cramer/issues/new). We appreciate your feedback and contributions!

## Additional Resources

- [Contributing to a Project on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
- [Forking Projects](https://guides.github.com/activities/forking/)
- [Creating a Pull Request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)

Thank you for your contributions!
