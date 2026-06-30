# Contributing

> **Note:** This repository is under double-blind review. Contributions are paused until the review period concludes. The information below describes the contribution workflow for the post-review release.

## Code Style

- Python 3.9+
- Follow PEP 8; line length 100 characters
- Use type hints for public function signatures
- Docstrings in Google or NumPy format

## Testing

```bash
pytest tests/ -v
```

Run the smoke test before submitting changes:
```bash
python scripts/run_smoke_test.py
```

## Anonymity Requirements

Until the review period concludes, contributors must ensure:

1. No author names in commit messages or comments
2. No institutional email addresses
3. No identifying file paths (use `<REDACTED>` placeholders if needed)
4. No usernames in any file
5. Use anonymous Git configuration when committing:
   ```bash
   git config user.name "Anonymous"
   git config user.email "anonymous@example.com"
   ```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Reporting Issues

Use the issue tracker to report bugs, propose features, or ask reproducibility questions. Include:
- Operating system and version
- Python and TensorFlow versions
- Minimal reproduction example
- Expected vs. observed behaviour
