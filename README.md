# barcode_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Learn Flutter](https://docs.flutter.dev/get-started/learn-flutter)
- [Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Flutter learning resources](https://docs.flutter.dev/reference/learning-resources)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Backend Run Instructions

1. Install Python dependencies from project root:

```bash
pip install -r requirements.txt
```

2. Install the `zbar` system library:

Ubuntu:

```bash
sudo apt install libzbar0
```

Mac:

```bash
brew install zbar
```

3. Run backend server:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
